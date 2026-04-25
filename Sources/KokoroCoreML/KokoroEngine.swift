@preconcurrency import AVFoundation
import Accelerate
import CoreML
import Foundation
import os

/// High-quality text-to-speech engine using Kokoro-82M CoreML models.
///
/// Uses a single dynamic frontend (predictor, CPU) + backend (decoder, GPU)
/// CoreML model pair. Accepts any token count up to 512 — no fixed buckets.
///
/// ```swift
/// let engine = try KokoroEngine(modelDirectory: myModelPath)
/// let result = try engine.synthesize(text: "Hello world", voice: "af_heart")
/// // result.samples contains 24kHz mono PCM float audio
/// ```
public final class KokoroEngine: @unchecked Sendable {

    // MARK: - Model Management

    /// Default directory for model storage.
    public static var defaultModelDirectory: URL {
        ModelManager.defaultDirectory()
    }

    /// Whether models are downloaded at the default directory.
    public static var isDownloaded: Bool {
        ModelManager.modelsAvailable(at: defaultModelDirectory)
    }

    /// Whether models are downloaded at a specific directory.
    public static func isDownloaded(at directory: URL) -> Bool {
        ModelManager.modelsAvailable(at: directory)
    }

    /// Download models to the default directory.
    public static func download(
        progress: (@Sendable (Double) -> Void)? = nil
    ) throws {
        try Self.download(to: defaultModelDirectory, progress: progress)
    }

    /// Download models to a specific directory.
    public static func download(
        to directory: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) throws {
        try ModelDownloader.download(to: directory, progress: progress)
    }

    // MARK: - Constants

    /// Output sample rate in Hz (24kHz).
    public static let sampleRate = 24_000

    /// Audio samples per duration frame.
    static let hopSize = 600

    /// Valid speed range for synthesis.
    static let speedRange: ClosedRange<Float> = 0.5...2.0

    /// Maximum token count the model accepts.
    static let maxTokens = 512

    /// Number of random phase channels for the iSTFTNet vocoder.
    private static let numPhases = 9

    /// Style content dimension (first half of the full 256-dim style vector).
    private static let sContentDim = VoiceStore.styleDim / 2

    /// Safety margin subtracted from max tokens when chunking phonemes.
    private static let tokenPadding = 7

    /// Silence samples inserted between chunks (100ms at 24kHz).
    private static let interChunkSilence = 2400

    /// Minimum audio to keep before the first non-BOS token when trimming BOS output.
    private static let minLeadingBOSPreroll = 960  // 40ms at 24kHz

    /// Fast-speech floor for adaptive BOS preroll.
    private static let minFastLeadingBOSPreroll = 720  // 30ms at 24kHz

    /// Maximum audio to keep before the first non-BOS token when trimming BOS output.
    private static let maxLeadingBOSPreroll = 2040  // 85ms at 24kHz

    /// Tail of the BOS span scanned for onset energy.
    private static let leadingBOSSearchWindow = 3000  // 125ms at 24kHz

    /// RMS analysis window for adaptive BOS trimming.
    private static let leadingBOSAnalysisWindow = 120  // 5ms at 24kHz

    /// Small margin kept before detected onset so fade-in does not eat the transient.
    private static let leadingBOSOnsetMargin = 120  // 5ms at 24kHz

    private enum Feature {
        static let inputIds = "input_ids"
        static let attentionMask = "attention_mask"
        static let refS = "ref_s"
        static let randomPhases = "random_phases"
        static let audio = "audio"
        static let audioLength = "audio_length_samples"
        static let predDurClamped = "pred_dur_clamped"
        static let speed = "speed"
        static let asr = "asr"
        static let f0Pred = "F0_pred"
        static let nPred = "N_pred"
        static let sContent = "s_content"
        static let har = "har"
    }

    private static let logger = Logger(
        subsystem: "com.kokorocoreml", category: "KokoroEngine")

    private let phonemizer: any Phonemizer
    private let g2pLock = NSLock()
    private let synthesizeLock = NSLock()
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private var _frontend: MLModel?
    private var _backend: MLModel?
    private var _loadError: Error?
    private let _loadCondition = NSCondition()
    private let _isReady = OSAllocatedUnfairLock(initialState: false)

    /// Whether the engine has completed background warmup.
    public var isReady: Bool { _isReady.withLock { $0 } }

    /// Creates a KokoroEngine from cached models.
    ///
    /// Returns immediately — models load and warm up in the background.
    /// Synthesis calls block until loading completes. Check ``isReady``
    /// to know when warmup is fully done.
    /// When `true`, forces all CoreML models to run on CPU only.
    /// Required for iOS Simulator where GPU compute produces garbage output.
    private let forceCPU: Bool

    public init(
        modelDirectory: URL,
        phonemizer: (any Phonemizer)? = nil,
        forceCPU: Bool = false
    ) throws {
        self.forceCPU = forceCPU
        guard ModelManager.modelsAvailable(at: modelDirectory) else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        self.phonemizer = phonemizer ?? EnglishG2P(british: false)

        let vocabURL = modelDirectory.appendingPathComponent("vocab_index.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            self.tokenizer = try Tokenizer.load(from: vocabURL)
        } else {
            self.tokenizer = try Tokenizer.loadFromBundle()
        }

        self.voiceStore = try VoiceStore(
            directory: modelDirectory.appendingPathComponent("voices"))

        let fePath = modelDirectory.appendingPathComponent("kokoro_frontend.mlmodelc")
        let bePath = modelDirectory.appendingPathComponent("kokoro_backend.mlmodelc")

        Self.logger.info("Loading dynamic frontend+backend (max \(Self.maxTokens) tokens)")

        let engine = self
        let thread = Thread {
            engine.loadAndWarmUp(frontendURL: fePath, backendURL: bePath)
        }
        thread.stackSize = 8 * 1024 * 1024
        thread.start()
    }

    // MARK: - Synthesis

    /// Synthesize text to PCM audio samples.
    public func synthesize(
        text: String, voice: String, speed: Float = 1.0
    ) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let clampedSpeed = Self.clampSpeed(speed)
        let (fullPhonemes, mergedIds) = prepareChunks(text: text)

        return try synthesizeTokens(
            phonemes: fullPhonemes, mergedIds: mergedIds, voice: voice,
            speed: clampedSpeed, startTime: t0)
    }

    /// Synthesize pre-phonemized IPA text to PCM audio samples.
    public func synthesize(
        ipa: String, voice: String, speed: Float = 1.0
    ) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let clampedSpeed = Self.clampSpeed(speed)
        let mergedIds = chunkAndTokenize(ipa)

        return try synthesizeTokens(
            phonemes: ipa, mergedIds: mergedIds, voice: voice,
            speed: clampedSpeed, startTime: t0)
    }

    /// Synthesize text using a raw 256-dim style vector instead of a named voice.
    ///
    /// The same style vector is used for all chunks (no length-indexed lookup).
    /// Use this for voice cloning where you have a custom blended embedding.
    public func synthesize(
        text: String, styleVector: [Float], speed: Float = 1.0
    ) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let clampedSpeed = Self.clampSpeed(speed)
        let (fullPhonemes, mergedIds) = prepareChunks(text: text)

        return try synthesizeTokensWithEmbedding(
            phonemes: fullPhonemes, mergedIds: mergedIds,
            styleVector: styleVector, speed: clampedSpeed, startTime: t0)
    }

    private func synthesizeTokens(
        phonemes: String, mergedIds: [[Int]], voice: String,
        speed: Float, startTime: CFAbsoluteTime
    ) throws -> SynthesisResult {

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for tokenIds in mergedIds {
            totalTokens += tokenIds.count

            let styleVector = try voiceStore.embedding(
                for: voice, tokenCount: tokenIds.count - 2)

            var (samples, durations) = try synthesizeChunk(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed)

            Self.applyFades(&samples)
            if !allSamples.isEmpty {
                allSamples.append(
                    contentsOf: [Float](repeating: 0, count: Self.interChunkSilence))
            }
            allSamples.append(contentsOf: samples)
            allDurations.append(contentsOf: durations)
        }

        Self.postProcess(&allSamples)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return SynthesisResult(
            samples: allSamples, phonemes: phonemes,
            tokenDurations: allDurations, tokenCount: totalTokens,
            synthesisTime: elapsed)
    }

    private func synthesizeTokensWithEmbedding(
        phonemes: String, mergedIds: [[Int]], styleVector: [Float],
        speed: Float, startTime: CFAbsoluteTime
    ) throws -> SynthesisResult {

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0

        for tokenIds in mergedIds {
            totalTokens += tokenIds.count

            var (samples, durations) = try synthesizeChunk(
                tokenIds: tokenIds, styleVector: styleVector, speed: speed)

            Self.applyFades(&samples)
            if !allSamples.isEmpty {
                allSamples.append(
                    contentsOf: [Float](repeating: 0, count: Self.interChunkSilence))
            }
            allSamples.append(contentsOf: samples)
            allDurations.append(contentsOf: durations)
        }

        Self.postProcess(&allSamples)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return SynthesisResult(
            samples: allSamples, phonemes: phonemes,
            tokenDurations: allDurations, tokenCount: totalTokens,
            synthesisTime: elapsed)
    }

    /// Apply fade-in and fade-out to suppress transients.
    private static func applyFades(_ samples: inout [Float]) {
        guard !samples.isEmpty else { return }
        samples.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!
            let fadeIn = min(120, buf.count)
            var ramp: Float = 0
            var step = 1.0 / Float(fadeIn)
            vDSP_vrampmul(ptr, 1, &ramp, &step, ptr, 1, vDSP_Length(fadeIn))
            let fadeOut = min(1200, buf.count)
            let fadeStart = buf.count - fadeOut
            ramp = 1.0
            step = -1.0 / Float(fadeOut)
            vDSP_vrampmul(
                ptr + fadeStart, 1, &ramp, &step, ptr + fadeStart, 1, vDSP_Length(fadeOut))
        }
    }

    private static let eqCoeffs: (nb0: Float, nb1: Float, nb2: Float, na1: Float, na2: Float) = {
        let fc: Float = 2500.0
        let fs = Float(sampleRate)
        let gain = powf(10.0, 2.0 / 40.0)
        let w0 = 2.0 * Float.pi * fc / fs
        let sinW0 = sinf(w0)
        let cosW0 = cosf(w0)
        let alpha = sinW0 / (2.0 * 0.8)
        let a0 = 1.0 + alpha / gain
        return (
            nb0: (1.0 + alpha * gain) / a0,
            nb1: (-2.0 * cosW0) / a0,
            nb2: (1.0 - alpha * gain) / a0,
            na1: (-2.0 * cosW0) / a0,
            na2: (1.0 - alpha / gain) / a0
        )
    }()

    private static let hpAlpha: Float = 1.0 - (2.0 * .pi * 80.0 / Float(sampleRate))

    /// Post-process audio in-place: high-pass filter, presence boost, peak normalize.
    private static func postProcess(_ samples: inout [Float]) {
        guard samples.count > 1 else { return }

        let alpha = hpAlpha
        let prechargeCount = min(240, samples.count)

        var prev: Float = 0
        var prevOut: Float = 0
        for i in stride(from: prechargeCount - 1, through: 0, by: -1) {
            let x = samples[i]
            prevOut = (x - prev) + alpha * prevOut
            prev = x
        }
        for i in 0..<samples.count {
            let x = samples[i]
            prevOut = (x - prev) + alpha * prevOut
            prev = x
            samples[i] = prevOut
        }

        let c = eqCoeffs
        var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
        for i in stride(from: prechargeCount - 1, through: 0, by: -1) {
            let x = samples[i]
            let y = c.nb0 * x + c.nb1 * x1 + c.nb2 * x2 - c.na1 * y1 - c.na2 * y2
            x2 = x1; x1 = x
            y2 = y1; y1 = y
        }
        for i in 0..<samples.count {
            let x = samples[i]
            let y = c.nb0 * x + c.nb1 * x1 + c.nb2 * x2 - c.na1 * y1 - c.na2 * y2
            x2 = x1; x1 = x
            y2 = y1; y1 = y
            samples[i] = y
        }

        var peak: Float = 0
        vDSP_maxmgv(samples, 1, &peak, vDSP_Length(samples.count))
        if peak > 0.001 {
            var scale = Float(0.95) / peak
            vDSP_vsmul(samples, 1, &scale, &samples, 1, vDSP_Length(samples.count))
        }
    }

    // MARK: - Voices

    /// Available voice preset names.
    public var availableVoices: [String] {
        voiceStore.availableVoices
    }

    /// All voice names with their generic 256-dim style embeddings.
    ///
    /// Use for voice cloning: compare user audio against each embedding's
    /// synthesized output to find the closest starting point.
    public var voiceEmbeddings: [(name: String, embedding: [Float])] {
        voiceStore.allGenericEmbeddings()
    }

    /// Register a custom voice at runtime.
    ///
    /// The embedding is immediately available for synthesis by name.
    /// Does not persist to disk — use ``VoiceStore`` binary format for that.
    public func registerVoice(name: String, embedding: [Float]) {
        voiceStore.registerVoice(name: name, embedding: embedding)
    }

    // MARK: - Loading & Warmup

    /// Block until both models are loaded, or throw if loading failed.
    private func awaitModels() throws -> (frontend: MLModel, backend: MLModel) {
        _loadCondition.lock()
        defer { _loadCondition.unlock() }
        while _frontend == nil || _backend == nil {
            if let error = _loadError { throw error }
            _loadCondition.wait()
        }
        return (_frontend!, _backend!)
    }

    /// Load both models in parallel and warm each up.
    ///
    /// Timeline:
    /// ```
    /// Thread A: fe_load → fe_warmup ──────────── be_warmup → isReady
    /// Thread B: be_load (GPU shader compile) ──┘
    /// ```
    /// Frontend warmup overlaps with backend GPU shader compilation.
    private func loadAndWarmUp(frontendURL: URL, backendURL: URL) {
        let feConfig = MLModelConfiguration()
        feConfig.computeUnits = .cpuOnly
        let beConfig = MLModelConfiguration()
        beConfig.computeUnits = forceCPU ? .cpuOnly : .all

        // Start backend on a separate thread (GPU shader compilation is the slow part)
        var loadedBE: MLModel?
        var beError: Error?
        let beSem = DispatchSemaphore(value: 0)

        let beThread = Thread {
            do {
                loadedBE = try MLModel(contentsOf: backendURL, configuration: beConfig)
            } catch {
                beError = error
            }
            beSem.signal()
        }
        beThread.stackSize = 8 * 1024 * 1024
        beThread.start()

        // Load frontend on this thread
        let loadedFE: MLModel
        do {
            loadedFE = try MLModel(contentsOf: frontendURL, configuration: feConfig)
        } catch {
            beSem.wait()
            _loadCondition.lock()
            _loadError = error
            _loadCondition.broadcast()
            _loadCondition.unlock()
            return
        }

        // Frontend loaded — warm it while backend is still compiling
        let feWarmupOutput = warmUpFrontend(loadedFE)

        // Wait for backend
        beSem.wait()

        guard let be = loadedBE else {
            _loadCondition.lock()
            _loadError = beError ?? KokoroError.inferenceFailed("Backend load failed")
            _loadCondition.broadcast()
            _loadCondition.unlock()
            return
        }

        // Publish models — unblocks any waiting synthesize() calls
        _loadCondition.lock()
        _frontend = loadedFE
        _backend = be
        _loadCondition.broadcast()
        _loadCondition.unlock()

        Self.logger.info("Models loaded, warming up backend...")

        // Warm backend with real frontend output
        if let output = feWarmupOutput {
            warmUpBackend(be, frontendOutput: output)
        }

        _isReady.withLock { $0 = true }
        Self.logger.info("Engine ready")
    }

    private func warmUpFrontend(_ model: MLModel) -> MLFeatureProvider? {
        do {
            let n = 4
            let ids = [0, 50, 1, 0]
            let inputIds = try MLMultiArray(shape: [1, n as NSNumber], dataType: .int32)
            let mask = try MLMultiArray(shape: [1, n as NSNumber], dataType: .int32)
            let refS = try MLMultiArray(
                shape: [1, VoiceStore.styleDim as NSNumber], dataType: .float32)
            let phases = try MLMultiArray(
                shape: [1, Self.numPhases as NSNumber], dataType: .float32)
            let speed = try MLMultiArray(shape: [1], dataType: .float32)

            for i in 0..<n { inputIds[i] = ids[i] as NSNumber; mask[i] = 1 }
            speed[0] = 1.0

            let input = try MLDictionaryFeatureProvider(dictionary: [
                Feature.inputIds: MLFeatureValue(multiArray: inputIds),
                Feature.attentionMask: MLFeatureValue(multiArray: mask),
                Feature.refS: MLFeatureValue(multiArray: refS),
                Feature.speed: MLFeatureValue(multiArray: speed),
                Feature.randomPhases: MLFeatureValue(multiArray: phases),
            ])
            return try model.prediction(from: input)
        } catch {
            Self.logger.warning("Frontend warmup failed (non-fatal): \(error.localizedDescription)")
            return nil
        }
    }

    private func warmUpBackend(_ model: MLModel, frontendOutput: MLFeatureProvider) {
        do {
            guard let asr = frontendOutput.featureValue(for: Feature.asr)?.multiArrayValue,
                let f0Pred = frontendOutput.featureValue(for: Feature.f0Pred)?.multiArrayValue,
                let nPred = frontendOutput.featureValue(for: Feature.nPred)?.multiArrayValue,
                let har = frontendOutput.featureValue(for: Feature.har)?.multiArrayValue
            else { return }

            let sContent = try MLMultiArray(
                shape: [1, Self.sContentDim as NSNumber], dataType: .float32)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                Feature.asr: MLFeatureValue(multiArray: asr),
                Feature.f0Pred: MLFeatureValue(multiArray: f0Pred),
                Feature.nPred: MLFeatureValue(multiArray: nPred),
                Feature.sContent: MLFeatureValue(multiArray: sContent),
                Feature.har: MLFeatureValue(multiArray: har),
            ])
            _ = try model.prediction(from: input)
        } catch {
            Self.logger.warning("Backend warmup failed (non-fatal): \(error.localizedDescription)")
        }
    }

    // MARK: - Private

    private static func clampSpeed(_ speed: Float) -> Float {
        min(max(speed, speedRange.lowerBound), speedRange.upperBound)
    }

    private func prepareChunks(text: String) -> (phonemes: String, mergedTokenIds: [[Int]]) {
        let paragraphs = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let fullPhonemes: String
        if paragraphs.count <= 1 {
            fullPhonemes = lockedPhonemize(text)
        } else {
            fullPhonemes = paragraphs.map { lockedPhonemize($0) }.joined(separator: " ")
        }
        return (fullPhonemes, chunkAndTokenize(fullPhonemes))
    }

    private func chunkAndTokenize(_ phonemes: String) -> [[Int]] {
        let chunks = Self.chunkPhonemes(
            phonemes, maxPhonemes: Self.maxTokens - Self.tokenPadding)
        let tokenized = chunks.map { tokenizer.encode($0) }

        var mergedIds: [[Int]] = []
        var currentIds: [Int] = []
        for ids in tokenized {
            let combined =
                currentIds.isEmpty
                ? ids
                : Array(currentIds.dropLast()) + Array(ids.dropFirst())
            if combined.count <= Self.maxTokens {
                currentIds = combined
            } else {
                if !currentIds.isEmpty { mergedIds.append(currentIds) }
                currentIds = ids
            }
        }
        if !currentIds.isEmpty { mergedIds.append(currentIds) }
        return mergedIds
    }

    private func lockedPhonemize(_ text: String) -> String {
        g2pLock.lock()
        defer { g2pLock.unlock() }
        return phonemizer.phonemize(text)
    }

    static func chunkPhonemes(_ phonemes: String, maxPhonemes: Int) -> [String] {
        guard phonemes.count > maxPhonemes else { return [phonemes] }

        let waterfallSets: [Set<Character>] = [
            Set("!.?\u{2026}"),
            Set(":;"),
            Set(",\u{2014}"),
        ]

        var chunks: [String] = []
        var remaining = phonemes[...]

        while remaining.count > maxPhonemes {
            let window = remaining.prefix(maxPhonemes)
            var splitIndex: String.Index?

            for punctSet in waterfallSets {
                if let idx = window.lastIndex(where: { punctSet.contains($0) }) {
                    splitIndex = window.index(after: idx)
                    break
                }
            }

            if splitIndex == nil {
                if let idx = window.lastIndex(of: " ") {
                    splitIndex = window.index(after: idx)
                }
            }

            let cut = splitIndex ?? window.endIndex
            let chunk = String(remaining[remaining.startIndex..<cut])
                .trimmingCharacters(in: .whitespaces)
            if !chunk.isEmpty { chunks.append(chunk) }
            remaining = remaining[cut...]
        }

        let tail = String(remaining).trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { chunks.append(tail) }

        return chunks
    }

    /// Synthesize a single chunk of token IDs.
    ///
    /// Creates input tensors at the exact token count — no padding.
    /// Dynamic CoreML model handles variable-length inputs.
    private func synthesizeChunk(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float
    ) throws -> (samples: [Float], durations: [Int]) {
        guard tokenIds.count <= Self.maxTokens else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: Self.maxTokens)
        }

        let (frontend, backend) = try awaitModels()

        synthesizeLock.lock()
        defer { synthesizeLock.unlock() }

        let n = tokenIds.count

        // Create tensors at exact token count — no padding
        let inputIds = try MLMultiArray(shape: [1, n as NSNumber], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, n as NSNumber], dataType: .int32)
        let refS = try MLMultiArray(
            shape: [1, VoiceStore.styleDim as NSNumber], dataType: .float32)
        let sContent = try MLMultiArray(
            shape: [1, Self.sContentDim as NSNumber], dataType: .float32)
        let randomPhases = try MLMultiArray(
            shape: [1, Self.numPhases as NSNumber], dataType: .float32)
        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

        MLArrayHelpers.fillTokenInputs(
            from: tokenIds, into: inputIds, mask: mask, maxLength: n)
        MLArrayHelpers.fillStyleArray(from: styleVector, into: refS)
        speedArray[0] = speed as NSNumber

        let phasePtr = randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        // --- Frontend (CPU) ---
        let frontendInput = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds: MLFeatureValue(multiArray: inputIds),
            Feature.attentionMask: MLFeatureValue(multiArray: mask),
            Feature.refS: MLFeatureValue(multiArray: refS),
            Feature.speed: MLFeatureValue(multiArray: speedArray),
            Feature.randomPhases: MLFeatureValue(multiArray: randomPhases),
        ])

        let feOutput = try frontend.prediction(from: frontendInput)

        guard let asr = feOutput.featureValue(for: Feature.asr)?.multiArrayValue,
            let f0Pred = feOutput.featureValue(for: Feature.f0Pred)?.multiArrayValue,
            let nPred = feOutput.featureValue(for: Feature.nPred)?.multiArrayValue,
            let har = feOutput.featureValue(for: Feature.har)?.multiArrayValue
        else {
            throw KokoroError.inferenceFailed("Missing frontend outputs")
        }

        // --- Backend (GPU) ---
        MLArrayHelpers.fillStyleArray(from: styleVector, into: sContent, dim: Self.sContentDim)

        let backendInput = try MLDictionaryFeatureProvider(dictionary: [
            Feature.asr: MLFeatureValue(multiArray: asr),
            Feature.f0Pred: MLFeatureValue(multiArray: f0Pred),
            Feature.nPred: MLFeatureValue(multiArray: nPred),
            Feature.sContent: MLFeatureValue(multiArray: sContent),
            Feature.har: MLFeatureValue(multiArray: har),
        ])

        let beOutput = try backend.prediction(from: backendInput)

        guard let audio = beOutput.featureValue(for: Feature.audio)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        // Durations and audio length from frontend
        let durations: [Int]
        if let predDur = feOutput.featureValue(for: Feature.predDurClamped)?.multiArrayValue {
            durations = (0..<min(predDur.count, tokenIds.count)).map { predDur[$0].intValue }
        } else {
            durations = []
        }

        let validSamples: Int
        if let lengthArray = feOutput.featureValue(for: Feature.audioLength)?.multiArrayValue,
            lengthArray[0].intValue > 0, lengthArray[0].intValue <= audio.count
        {
            validSamples = lengthArray[0].intValue
        } else if !durations.isEmpty {
            let totalFrames = durations.reduce(0, +)
            validSamples = min(totalFrames * Self.hopSize + 600, audio.count)
        } else {
            validSamples = audio.count
        }

        var samples = MLArrayHelpers.extractFloats(from: audio, maxCount: validSamples)

        if let leadingFrames = durations.first, leadingFrames > 0 {
            let leadSamples = min(leadingFrames * Self.hopSize, samples.count)
            let trimSamples = Self.adaptiveLeadingBOSTrimSamples(
                in: samples, leadSamples: leadSamples, speed: speed)
            if trimSamples > 0 {
                samples.removeFirst(trimSamples)
            }
        }

        Self.removeTailArtifact(&samples)
        return (samples, durations)
    }

    /// Choose how much of the leading BOS span to remove without cutting first-phoneme onset.
    static func adaptiveLeadingBOSTrimSamples(
        in samples: [Float], leadSamples: Int, speed: Float
    ) -> Int {
        guard leadSamples > 0 else { return 0 }

        let clampedLeadSamples = min(leadSamples, samples.count)
        let speedScale = 1.0 / max(1.0, speed)
        let scaledMinPreroll = max(
            minFastLeadingBOSPreroll,
            Int((Float(minLeadingBOSPreroll) * speedScale).rounded()))
        let scaledMaxPreroll = max(
            scaledMinPreroll,
            Int((Float(maxLeadingBOSPreroll) * speedScale).rounded()))

        let minPreroll = min(scaledMinPreroll, clampedLeadSamples)
        let maxPreroll = min(scaledMaxPreroll, clampedLeadSamples)
        let fallbackPreroll = min(leadingBOSOnsetMargin, clampedLeadSamples)
        let searchStart = max(0, clampedLeadSamples - leadingBOSSearchWindow)
        let window = leadingBOSAnalysisWindow
        let analysisEnd = min(samples.count, clampedLeadSamples + 2 * window)
        guard analysisEnd - searchStart >= window else {
            return max(0, clampedLeadSamples - fallbackPreroll)
        }

        var rmsWindows: [(offset: Int, rms: Float)] = []
        var offset = searchStart
        while offset + window <= analysisEnd {
            var sumSquares: Float = 0
            for i in offset..<(offset + window) {
                let sample = samples[i]
                sumSquares += sample * sample
            }
            let rms = sqrtf(sumSquares / Float(window))
            rmsWindows.append((offset, rms))
            offset += window
        }

        let bosWindows = rmsWindows.filter { $0.offset < clampedLeadSamples }
        let maxBOSRMS = bosWindows.reduce(Float(0)) { max($0, $1.rms) }
        guard maxBOSRMS > 0 else {
            return max(0, clampedLeadSamples - fallbackPreroll)
        }

        let baselineCount = min(5, bosWindows.count)
        let baselineRMS =
            bosWindows.prefix(baselineCount).reduce(Float(0)) { $0 + $1.rms }
            / Float(baselineCount)
        let threshold = max(maxBOSRMS * 0.10, baselineRMS * 8.0, Float(0.00002))

        var onsetOffset: Int?
        for index in rmsWindows.indices {
            guard rmsWindows[index].offset < clampedLeadSamples else { break }
            guard rmsWindows[index].rms >= threshold else { continue }

            let lookaheadEnd = min(index + 5, rmsWindows.count)
            var sustainedWindows = 0
            for lookahead in index..<lookaheadEnd {
                if rmsWindows[lookahead].rms >= threshold {
                    sustainedWindows += 1
                }
            }

            if sustainedWindows >= 3 {
                onsetOffset = rmsWindows[index].offset
                break
            }
        }

        let preroll: Int
        if let onsetOffset {
            let detectedPreroll = clampedLeadSamples - onsetOffset + leadingBOSOnsetMargin
            preroll = min(maxPreroll, max(minPreroll, detectedPreroll))
        } else {
            preroll = fallbackPreroll
        }

        return max(0, clampedLeadSamples - preroll)
    }

    /// Remove spurious spike artifacts from the tail of synthesized audio.
    private static func removeTailArtifact(_ samples: inout [Float]) {
        let tailLength = min(samples.count, Int(0.1 * Float(sampleRate)))
        let window = sampleRate / 1000
        let tailStart = samples.count - tailLength

        var rmsWindows: [(index: Int, rms: Float)] = []
        var i = tailStart
        while i + window <= samples.count {
            var sum: Float = 0
            for j in i..<(i + window) {
                sum += samples[j] * samples[j]
            }
            let rms = (sum / Float(window)).squareRoot()
            rmsWindows.append((i, rms))
            i += window
        }

        guard !rmsWindows.isEmpty else { return }

        let spikeThreshold: Float = 0.0015
        let quietThreshold: Float = 0.0006

        var spikeIdx: Int?
        for idx in stride(from: rmsWindows.count - 1, through: 0, by: -1) {
            if rmsWindows[idx].rms > spikeThreshold {
                spikeIdx = idx
                break
            }
        }

        guard let spike = spikeIdx else { return }

        var gapIdx: Int?
        for idx in stride(from: spike, through: 0, by: -1) {
            if rmsWindows[idx].rms < quietThreshold {
                gapIdx = idx
                break
            }
        }

        guard let gap = gapIdx else { return }

        let zeroFrom = rmsWindows[gap].index
        let fadeLen = 2 * window
        for j in zeroFrom..<min(zeroFrom + fadeLen, samples.count) {
            let progress = Float(j - zeroFrom) / Float(fadeLen)
            samples[j] *= (1.0 - progress)
        }
        for j in (zeroFrom + fadeLen)..<samples.count {
            samples[j] = 0
        }
    }

    // MARK: - Streaming

    /// Audio format for streaming buffers (24kHz, mono, float32).
    public static let audioFormat = AVAudioFormat(
        standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

    /// Stream synthesized audio as playback-ready buffers.
    public func speak(
        _ text: String,
        voice: String,
        speed: Float = 1.0
    ) throws -> AsyncStream<SpeakEvent> {
        guard availableVoices.contains(voice) else {
            throw KokoroError.voiceNotFound(voice)
        }

        let clampedSpeed = Self.clampSpeed(speed)
        let (_, mergedIds) = prepareChunks(text: text)
        guard !mergedIds.isEmpty else { return AsyncStream { $0.finish() } }

        return AsyncStream { continuation in
            let thread = Thread {
                let format = Self.audioFormat

                for tokenIds in mergedIds {
                    if Thread.current.isCancelled { break }

                    do {
                        let styleVector = try self.voiceStore.embedding(
                            for: voice, tokenCount: tokenIds.count - 2)

                        var (samples, _) = try self.synthesizeChunk(
                            tokenIds: tokenIds, styleVector: styleVector,
                            speed: clampedSpeed)
                        Self.applyFades(&samples)
                        Self.postProcess(&samples)

                        if let buffer = Self.makePCMBuffer(from: samples, format: format) {
                            continuation.yield(.audio(buffer))
                        }
                    } catch {
                        Self.logger.error(
                            "Streaming chunk failed: \(error.localizedDescription)")
                        continuation.yield(.chunkFailed(error))
                    }
                }

                continuation.finish()
            }
            thread.stackSize = 8 * 1024 * 1024
            nonisolated(unsafe) let unsafeThread = thread
            continuation.onTermination = { _ in unsafeThread.cancel() }
            thread.start()
        }
    }

    /// Stream synthesized audio using a raw 256-dim style vector.
    public func speak(
        _ text: String,
        styleVector: [Float],
        speed: Float = 1.0
    ) throws -> AsyncStream<SpeakEvent> {
        let clampedSpeed = Self.clampSpeed(speed)
        let (_, mergedIds) = prepareChunks(text: text)
        guard !mergedIds.isEmpty else { return AsyncStream { $0.finish() } }

        return AsyncStream { continuation in
            let thread = Thread {
                let format = Self.audioFormat

                for tokenIds in mergedIds {
                    if Thread.current.isCancelled { break }

                    do {
                        var (samples, _) = try self.synthesizeChunk(
                            tokenIds: tokenIds, styleVector: styleVector,
                            speed: clampedSpeed)
                        Self.applyFades(&samples)
                        Self.postProcess(&samples)

                        if let buffer = Self.makePCMBuffer(from: samples, format: format) {
                            continuation.yield(.audio(buffer))
                        }
                    } catch {
                        Self.logger.error(
                            "Streaming chunk failed: \(error.localizedDescription)")
                        continuation.yield(.chunkFailed(error))
                    }
                }

                continuation.finish()
            }
            thread.stackSize = 8 * 1024 * 1024
            nonisolated(unsafe) let unsafeThread = thread
            continuation.onTermination = { _ in unsafeThread.cancel() }
            thread.start()
        }
    }

    /// Convert float samples to a playback-ready `AVAudioPCMBuffer`.
    public static func makePCMBuffer(
        from samples: [Float], format: AVAudioFormat
    ) -> AVAudioPCMBuffer? {
        guard !samples.isEmpty else { return nil }
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            guard let dst = buffer.floatChannelData?[0], let srcBase = src.baseAddress
            else { return }
            dst.update(from: srcBase, count: samples.count)
        }
        return buffer
    }

}

// MARK: - SpeakEvent

/// Events yielded by ``KokoroEngine/speak(_:voice:speed:)``.
public enum SpeakEvent: @unchecked Sendable {
    /// A playback-ready audio buffer for one synthesized chunk.
    case audio(AVAudioPCMBuffer)
    /// A chunk failed to synthesize. The stream continues with remaining chunks.
    case chunkFailed(any Error)
}

// MARK: - SynthesisResult

/// Result from a text-to-speech synthesis call.
public struct SynthesisResult: Sendable {
    /// 24kHz mono PCM float samples.
    public let samples: [Float]

    /// IPA phoneme string produced by the G2P pipeline.
    public let phonemes: String

    /// Per-token predicted durations in audio frames.
    let tokenDurations: [Int]

    /// Number of input tokens processed.
    public let tokenCount: Int

    /// Wall-clock synthesis time in seconds.
    public let synthesisTime: TimeInterval

    /// Audio duration in seconds.
    public var duration: TimeInterval {
        Double(samples.count) / Double(KokoroEngine.sampleRate)
    }

    /// Real-time factor (audio duration / synthesis time).
    public var realTimeFactor: Double {
        synthesisTime > 0 ? duration / synthesisTime : 0
    }

    init(
        samples: [Float],
        phonemes: String = "",
        tokenDurations: [Int] = [],
        tokenCount: Int,
        synthesisTime: TimeInterval
    ) {
        self.samples = samples
        self.phonemes = phonemes
        self.tokenDurations = tokenDurations
        self.tokenCount = tokenCount
        self.synthesisTime = synthesisTime
    }
}
