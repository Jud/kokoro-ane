import Accelerate
@preconcurrency import AVFoundation
import CoreML
import Foundation
import os

/// Model size bucket for automatic selection based on token count.
public enum ModelBucket: String, CaseIterable, Sendable, Comparable {
    /// Short utterances — 124 tokens max (~5s audio).
    case small
    /// Medium utterances — 242 tokens max (~10s audio).
    case medium

    /// CoreML model bundle name (without .mlmodelc extension).
    public var modelName: String {
        switch self {
        case .small: "kokoro_21_5s"
        case .medium: "kokoro_24_10s"
        }
    }

    /// Frontend model bundle name.
    public var frontendModelName: String { modelName + "_frontend" }

    /// Backend model bundle name.
    public var backendModelName: String { modelName + "_backend" }

    /// Maximum input token count for this bucket.
    public var maxTokens: Int {
        switch self {
        case .small: 124
        case .medium: 242
        }
    }

    /// Select the smallest bucket that fits the token count.
    /// Returns nil if no bucket is large enough.
    ///
    /// Assumes `available` is sorted ascending (which `activeBuckets` always is).
    public static func select(forTokenCount count: Int, available: [ModelBucket]) -> ModelBucket? {
        assert(available == available.sorted(), "available must be sorted ascending")
        return available.first { count <= $0.maxTokens }
    }

    public static func < (lhs: ModelBucket, rhs: ModelBucket) -> Bool {
        lhs.maxTokens < rhs.maxTokens
    }
}

/// High-quality text-to-speech engine using Kokoro-82M CoreML models.
///
/// Uses split frontend (predictor, CPU) + backend (decoder, ANE) CoreML
/// models per bucket size for fast Neural Engine inference.
///
/// ```swift
/// let engine = try KokoroEngine(modelDirectory: myModelPath)
/// engine.warmUp()
/// let result = try engine.synthesize(text: "Hello world", voice: "af_heart")
/// // result.samples contains 24kHz mono PCM float audio
/// ```
public final class KokoroEngine: @unchecked Sendable {
    // Immutable after init. EnglishG2P has internal mutable state,
    // so all phonemize calls are serialized through g2pLock.

    /// Output sample rate in Hz (24kHz).
    public static let sampleRate = 24_000

    /// Audio samples per duration frame (from model.mil: total_frames * 600).
    public static let hopSize = 600

    /// Valid speed range for synthesis.
    public static let speedRange: ClosedRange<Float> = 0.5...2.0

    /// Number of random phase channels for the iSTFTNet vocoder.
    private static let numPhases = 9

    /// Style content dimension (first half of the full 256-dim style vector).
    private static let sContentDim = VoiceStore.styleDim / 2

    /// Safety margin subtracted from max bucket tokens when chunking phonemes.
    private static let tokenPadding = 7

    /// Silence samples inserted between chunks (100ms at 24kHz).
    private static let interChunkSilence = 2400

    private enum Feature {
        static let inputIds = "input_ids"
        static let attentionMask = "attention_mask"
        static let refS = "ref_s"
        static let randomPhases = "random_phases"
        static let audio = "audio"
        static let audioLength = "audio_length_samples"
        static let predDurClamped = "pred_dur_clamped"
        static let predDur = "pred_dur"
        static let speed = "speed"
        static let asr = "asr"
        static let f0Pred = "F0_pred"
        static let nPred = "N_pred"
        static let sContent = "s_content"
        static let har = "har"
    }

    /// Per-bucket CoreML models and pre-allocated buffers.
    private struct BucketResources {
        let frontend: MLModel  // CPU_ONLY
        let backend: MLModel  // .all (ANE preferred)
        let inputIds: MLMultiArray
        let mask: MLMultiArray
        let refS: MLMultiArray
        let sContent: MLMultiArray
        let randomPhases: MLMultiArray
        let speed: MLMultiArray
    }

    private static let logger = Logger(
        subsystem: "com.kokorotts", category: "KokoroEngine")

    /// Threshold in seconds above which model.prediction() is considered slow
    /// (likely ANE fallback to CPU/GPU).
    private static let slowPredictionThreshold: TimeInterval = 0.150

    private let g2p: EnglishG2P
    private let g2pLock = NSLock()
    private let synthesizeLock = NSLock()
    private let tokenizer: Tokenizer
    private let voiceStore: VoiceStore
    private let bucketResources: [ModelBucket: BucketResources]

    /// Which buckets are loaded and available for synthesis (sorted ascending).
    public let activeBuckets: [ModelBucket]

    /// Creates a KokoroEngine from cached models.
    ///
    /// Loads all available model buckets from the directory. At least one
    /// `.mlmodelc` model bundle and a `voices/` directory must be present.
    ///
    /// - Parameter modelDirectory: Path containing `.mlmodelc` model bundles
    ///   and a `voices/` directory with voice embedding JSON files.
    /// - Throws: ``KokoroError/modelsNotAvailable(_:)`` if no models found.
    public init(modelDirectory: URL) throws {
        guard ModelManager.modelsAvailable(at: modelDirectory) else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        self.g2p = EnglishG2P(british: false)

        let vocabURL = modelDirectory.appendingPathComponent("vocab_index.json")
        if FileManager.default.fileExists(atPath: vocabURL.path) {
            self.tokenizer = try Tokenizer.load(from: vocabURL)
        } else {
            self.tokenizer = try Tokenizer.loadFromBundle()
        }

        self.voiceStore = try VoiceStore(directory: modelDirectory.appendingPathComponent("voices"))

        let feConfig = MLModelConfiguration()
        feConfig.computeUnits = .cpuOnly
        let beConfig = MLModelConfiguration()
        beConfig.computeUnits = .all

        var resources: [ModelBucket: BucketResources] = [:]
        for bucket in ModelBucket.allCases {
            let feURL = modelDirectory.appendingPathComponent(
                bucket.frontendModelName + ".mlmodelc")
            let beURL = modelDirectory.appendingPathComponent(
                bucket.backendModelName + ".mlmodelc")
            guard FileManager.default.fileExists(atPath: feURL.path),
                FileManager.default.fileExists(atPath: beURL.path)
            else { continue }
            do {
                let maxTokens = bucket.maxTokens

                let frontend = try MLModel(contentsOf: feURL, configuration: feConfig)
                let backend = try MLModel(contentsOf: beURL, configuration: beConfig)

                let res = BucketResources(
                    frontend: frontend,
                    backend: backend,
                    inputIds: try MLMultiArray(
                        shape: [1, maxTokens as NSNumber], dataType: .int32),
                    mask: try MLMultiArray(
                        shape: [1, maxTokens as NSNumber], dataType: .int32),
                    refS: try MLMultiArray(
                        shape: [1, VoiceStore.styleDim as NSNumber], dataType: .float32),
                    sContent: try MLMultiArray(
                        shape: [1, Self.sContentDim as NSNumber], dataType: .float32),
                    randomPhases: try MLMultiArray(
                        shape: [1, Self.numPhases as NSNumber], dataType: .float32),
                    speed: try MLMultiArray(shape: [1], dataType: .float32)
                )
                resources[bucket] = res
                Self.logger.info("Loaded \(bucket.modelName) frontend+backend (\(maxTokens) tokens)")
            } catch {
                Self.logger.warning("Failed to load \(bucket.modelName): \(error)")
            }
        }

        guard !resources.isEmpty else {
            throw KokoroError.modelsNotAvailable(modelDirectory)
        }

        self.bucketResources = resources
        self.activeBuckets = ModelBucket.allCases.filter { resources[$0] != nil }
    }

    // MARK: - Synthesis

    /// Synthesize text to PCM audio samples.
    ///
    /// Splits text into sentences, merges consecutive short ones that fit in a
    /// single bucket for better prosody, then synthesizes each chunk.
    ///
    /// - Parameters:
    ///   - text: Text to speak (any length).
    ///   - voice: Voice preset name (e.g. `"af_heart"`, `"am_adam"`).
    ///     See ``availableVoices`` for valid names.
    ///   - speed: Speech rate multiplier (0.5 = half speed, 2.0 = double speed).
    ///     Clamped to 0.5...2.0. Default is 1.0.
    ///   - rawAudio: If true, skip post-processing (HP filter, EQ, normalization).
    /// - Returns: Synthesis result with PCM samples and metadata.
    /// - Throws: ``KokoroError/voiceNotFound(_:)`` if the voice doesn't exist.
    public func synthesize(
        text: String, voice: String, speed: Float = 1.0,
        bucket: ModelBucket? = nil, rawAudio: Bool = false
    ) throws -> SynthesisResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let clampedSpeed = Self.clampSpeed(speed)
        let (fullPhonemes, mergedIds) = prepareChunks(text: text)

        var allSamples: [Float] = []
        var allDurations: [Int] = []
        var totalTokens = 0
        var selectedBucket: ModelBucket?

        for tokenIds in mergedIds {
            totalTokens += tokenIds.count

            let styleVector = try voiceStore.embedding(for: voice, tokenCount: tokenIds.count - 2)

            let useBucket: ModelBucket
            if let forced = bucket {
                useBucket = forced
            } else if let auto = ModelBucket.select(
                forTokenCount: tokenIds.count, available: activeBuckets)
            {
                useBucket = auto
            } else {
                useBucket = activeBuckets.last!
            }
            selectedBucket = useBucket

            var (samples, durations) = try synthesizeChunk(
                tokenIds: tokenIds, styleVector: styleVector, speed: clampedSpeed,
                bucket: useBucket)

            applyFades(&samples)
            if !allSamples.isEmpty {
                allSamples.append(contentsOf: [Float](repeating: 0, count: Self.interChunkSilence))
            }
            allSamples.append(contentsOf: samples)
            allDurations.append(contentsOf: durations)
        }

        if !rawAudio { postProcess(&allSamples) }

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        return SynthesisResult(
            samples: allSamples, phonemes: fullPhonemes,
            tokenDurations: allDurations, tokenCount: totalTokens,
            synthesisTime: elapsed, bucket: selectedBucket)
    }

    // MARK: - Voices

    /// Available voice preset names.
    public var availableVoices: [String] {
        voiceStore.availableVoices
    }

    // MARK: - Warmup

    /// Pre-warm the CoreML compilation cache for all loaded buckets.
    ///
    /// Triggers on-device model compilation for the Neural Engine. Call once
    /// at app startup to avoid a cold-start delay on the first synthesis call.
    /// Safe to skip — the first real synthesis will just be slower.
    public func warmUp() {
        for bucket in activeBuckets {
            do {
                let dummyTokens = [Int](repeating: 0, count: bucket.maxTokens)
                let dummyStyle = [Float](repeating: 0, count: VoiceStore.styleDim)
                _ = try synthesizeChunk(
                    tokenIds: dummyTokens, styleVector: dummyStyle, speed: 1.0,
                    bucket: bucket)
            } catch {
                Self.logger.warning(
                    "Warmup failed for \(bucket.modelName) (non-fatal): \(error.localizedDescription)"
                )
            }
        }
    }

    // MARK: - Private

    private static func clampSpeed(_ speed: Float) -> Float {
        min(max(speed, speedRange.lowerBound), speedRange.upperBound)
    }

    /// Phonemize, chunk, tokenize, and merge text into batches of token IDs.
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

        let maxTokens = activeBuckets.last!.maxTokens
        let chunks = Self.chunkPhonemes(fullPhonemes, maxPhonemes: maxTokens - Self.tokenPadding)
        let tokenized = chunks.map { tokenizer.encode($0) }

        var mergedIds: [[Int]] = []
        var currentIds: [Int] = []
        for ids in tokenized {
            let combined =
                currentIds.isEmpty
                ? ids
                : Array(currentIds.dropLast()) + Array(ids.dropFirst())
            if combined.count <= maxTokens {
                currentIds = combined
            } else {
                if !currentIds.isEmpty { mergedIds.append(currentIds) }
                currentIds = ids
            }
        }
        if !currentIds.isEmpty { mergedIds.append(currentIds) }

        return (fullPhonemes, mergedIds)
    }

    private func lockedPhonemize(_ text: String) -> String {
        g2pLock.lock()
        defer { g2pLock.unlock() }
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }

    /// Chunk a phoneme string into pieces that each fit within `maxPhonemes` characters.
    ///
    /// Uses a waterfall punctuation search matching the reference implementation:
    /// first try to split at sentence-ending punctuation (`!.?…`), then at
    /// clause boundaries (`:;`), then at phrase boundaries (`,—`), then at spaces.
    static func chunkPhonemes(_ phonemes: String, maxPhonemes: Int) -> [String] {
        guard phonemes.count > maxPhonemes else { return [phonemes] }

        let waterfallSets: [Set<Character>] = [
            Set("!.?\u{2026}"),  // sentence endings
            Set(":;"),  // clause boundaries
            Set(",\u{2014}"),  // phrase boundaries
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

    private func synthesizeChunk(
        tokenIds: [Int],
        styleVector: [Float],
        speed: Float,
        bucket: ModelBucket
    ) throws -> (samples: [Float], durations: [Int]) {
        guard let res = bucketResources[bucket] else {
            throw KokoroError.inferenceFailed("Bucket \(bucket.modelName) not loaded")
        }
        guard tokenIds.count <= bucket.maxTokens else {
            throw KokoroError.textTooLong(
                tokenCount: tokenIds.count, maxTokens: bucket.maxTokens)
        }

        synthesizeLock.lock()
        defer { synthesizeLock.unlock() }

        // --- Frontend: predictor + GeneratorFrontEnd (CPU) ---
        MLArrayHelpers.fillTokenInputs(
            from: tokenIds, into: res.inputIds, mask: res.mask, maxLength: bucket.maxTokens)
        MLArrayHelpers.fillStyleArray(from: styleVector, into: res.refS)
        res.speed[0] = speed as NSNumber

        let phasePtr = res.randomPhases.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<Self.numPhases {
            phasePtr[i] = Float.random(in: 0..<(2 * .pi))
        }

        let frontendInput = try MLDictionaryFeatureProvider(dictionary: [
            Feature.inputIds: MLFeatureValue(multiArray: res.inputIds),
            Feature.attentionMask: MLFeatureValue(multiArray: res.mask),
            Feature.refS: MLFeatureValue(multiArray: res.refS),
            Feature.speed: MLFeatureValue(multiArray: res.speed),
            Feature.randomPhases: MLFeatureValue(multiArray: res.randomPhases),
        ])

        let predictionStart = CFAbsoluteTimeGetCurrent()
        let feOutput = try res.frontend.prediction(from: frontendInput)

        guard let asr = feOutput.featureValue(for: Feature.asr)?.multiArrayValue,
            let f0Pred = feOutput.featureValue(for: Feature.f0Pred)?.multiArrayValue,
            let nPred = feOutput.featureValue(for: Feature.nPred)?.multiArrayValue,
            let har = feOutput.featureValue(for: Feature.har)?.multiArrayValue
        else {
            throw KokoroError.inferenceFailed("Missing frontend outputs")
        }

        // --- Backend: DecoderBackEnd (ANE, ~112ms, no atan2) ---
        let sContentPtr = res.sContent.dataPointer.assumingMemoryBound(to: Float.self)
        let sCount = min(styleVector.count, Self.sContentDim)
        styleVector.withUnsafeBufferPointer { src in
            sContentPtr.update(from: src.baseAddress!, count: sCount)
        }
        if sCount < Self.sContentDim {
            sContentPtr.advanced(by: sCount).update(repeating: 0, count: Self.sContentDim - sCount)
        }

        let backendInput = try MLDictionaryFeatureProvider(dictionary: [
            Feature.asr: MLFeatureValue(multiArray: asr),
            Feature.f0Pred: MLFeatureValue(multiArray: f0Pred),
            Feature.nPred: MLFeatureValue(multiArray: nPred),
            Feature.sContent: MLFeatureValue(multiArray: res.sContent),
            Feature.har: MLFeatureValue(multiArray: har),
        ])

        let beOutput = try res.backend.prediction(from: backendInput)
        let predictionElapsed = CFAbsoluteTimeGetCurrent() - predictionStart

        if predictionElapsed > Self.slowPredictionThreshold {
            let ms = String(format: "%.0f", predictionElapsed * 1_000)
            Self.logger.warning(
                "\(bucket.modelName) prediction took \(ms)ms (possible ANE fallback to CPU/GPU)")
        }

        guard let audio = beOutput.featureValue(for: Feature.audio)?.multiArrayValue else {
            throw KokoroError.inferenceFailed("Missing audio output")
        }

        // Durations and audio length come from frontend
        let durations: [Int]
        let predDur =
            feOutput.featureValue(for: Feature.predDurClamped)?.multiArrayValue
            ?? feOutput.featureValue(for: Feature.predDur)?.multiArrayValue
        if let predDur {
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

        let samples = MLArrayHelpers.extractFloats(from: audio, maxCount: validSamples)
        return (samples, durations)
    }

    /// Apply fade-in and fade-out to suppress transients.
    private func applyFades(_ samples: inout [Float]) {
        guard samples.count > 0 else { return }
        samples.withUnsafeMutableBufferPointer { buf in
            let ptr = buf.baseAddress!
            let fadeIn = min(120, buf.count)  // 5ms
            var ramp: Float = 0
            var step = 1.0 / Float(fadeIn)
            vDSP_vrampmul(ptr, 1, &ramp, &step, ptr, 1, vDSP_Length(fadeIn))
            let fadeOut = min(1200, buf.count)  // 50ms
            let fadeStart = buf.count - fadeOut
            ramp = 1.0
            step = -1.0 / Float(fadeOut)
            vDSP_vrampmul(ptr + fadeStart, 1, &ramp, &step, ptr + fadeStart, 1, vDSP_Length(fadeOut))
        }
    }

    // Precomputed biquad coefficients for presence boost EQ.
    // Peaking EQ: center=2.5kHz, Q=0.8, gain=+2dB at 24kHz sample rate.
    private static let eqCoeffs: (nb0: Float, nb1: Float, nb2: Float, na1: Float, na2: Float) = {
        let fc: Float = 2500.0
        let fs = Float(sampleRate)
        let A = powf(10.0, 2.0 / 40.0)  // +2dB
        let w0 = 2.0 * Float.pi * fc / fs
        let sinW0 = sinf(w0)
        let cosW0 = cosf(w0)
        let alpha = sinW0 / (2.0 * 0.8)  // Q=0.8
        let a0 = 1.0 + alpha / A
        return (
            nb0: (1.0 + alpha * A) / a0,
            nb1: (-2.0 * cosW0) / a0,
            nb2: (1.0 - alpha * A) / a0,
            na1: (-2.0 * cosW0) / a0,
            na2: (1.0 - alpha / A) / a0
        )
    }()

    // Precomputed high-pass filter coefficient: alpha = 1 - (2π * 80Hz / sampleRate)
    private static let hpAlpha: Float = 1.0 - (2.0 * .pi * 80.0 / Float(sampleRate))

    // MARK: - Streaming

    /// Audio format for streaming buffers (24kHz, mono, float32).
    ///
    /// Use this to configure your `AVAudioEngine` for playback:
    /// ```swift
    /// audioEngine.connect(playerNode, to: audioEngine.mainMixerNode,
    ///                     format: KokoroEngine.audioFormat)
    /// ```
    public static let audioFormat = AVAudioFormat(
        standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

    /// Stream synthesized audio as playback-ready buffers.
    ///
    /// Automatically chunks long text and yields each segment as an
    /// `AVAudioPCMBuffer` ready for `AVAudioPlayerNode.scheduleBuffer()`.
    /// Consumers just iterate and play — no manual chunking or PCM handling needed.
    ///
    /// ```swift
    /// for await buffer in try engine.speak("Long text...", voice: "af_heart") {
    ///     playerNode.scheduleBuffer(buffer)
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - text: Text to speak (any length — automatically chunked).
    ///   - voice: Voice preset name (e.g. `"af_heart"`).
    ///   - speed: Speech rate multiplier (0.5–2.0). Default 1.0.
    /// - Returns: Stream of playback-ready 24kHz mono audio buffers.
    /// - Throws: ``KokoroError/voiceNotFound(_:)`` if the voice doesn't exist.
    public func speak(
        _ text: String,
        voice: String,
        speed: Float = 1.0
    ) throws -> AsyncStream<AVAudioPCMBuffer> {
        guard availableVoices.contains(voice) else {
            throw KokoroError.voiceNotFound(voice)
        }

        let clampedSpeed = Self.clampSpeed(speed)

        return AsyncStream { continuation in
            let thread = Thread {
                let (_, mergedIds) = self.prepareChunks(text: text)
                guard !mergedIds.isEmpty else {
                    continuation.finish()
                    return
                }

                let format = Self.audioFormat
                var isFirst = true

                for tokenIds in mergedIds {
                    if Thread.current.isCancelled { break }

                    do {
                        let styleVector = try self.voiceStore.embedding(
                            for: voice, tokenCount: tokenIds.count - 2)
                        let bucket =
                            ModelBucket.select(
                                forTokenCount: tokenIds.count, available: self.activeBuckets)
                            ?? self.activeBuckets.last!

                        var (samples, _) = try self.synthesizeChunk(
                            tokenIds: tokenIds, styleVector: styleVector,
                            speed: clampedSpeed, bucket: bucket)

                        self.applyFades(&samples)
                        self.postProcess(&samples)

                        if !isFirst,
                            let gap = Self.makePCMBuffer(
                                from: [Float](repeating: 0, count: Self.interChunkSilence),
                                format: format)
                        {
                            continuation.yield(gap)
                        }

                        if let buffer = Self.makePCMBuffer(from: samples, format: format) {
                            continuation.yield(buffer)
                        }
                        isFirst = false
                    } catch {
                        Self.logger.error(
                            "Streaming chunk failed: \(error.localizedDescription)")
                    }
                }

                continuation.finish()
            }
            thread.stackSize = 8 * 1024 * 1024  // CoreML needs deep stacks
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

    /// Post-process audio in-place: high-pass filter, presence boost, peak normalize.
    private func postProcess(_ samples: inout [Float]) {
        guard samples.count > 1 else { return }

        let alpha = Self.hpAlpha
        let prechargeCount = min(240, samples.count)  // 10ms at 24kHz

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

        let c = Self.eqCoeffs

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
}

// MARK: - SynthesisResult

/// Result from a text-to-speech synthesis call.
public struct SynthesisResult: Sendable {
    /// 24kHz mono PCM float samples.
    public let samples: [Float]

    /// IPA phoneme string produced by the G2P pipeline.
    public let phonemes: String

    /// Per-token predicted durations in audio frames (at 24kHz).
    /// Useful for mapping playback position back to input tokens.
    public let tokenDurations: [Int]

    /// Number of input tokens processed.
    public let tokenCount: Int

    /// Wall-clock synthesis time in seconds.
    public let synthesisTime: TimeInterval

    /// Which model bucket was used (last sentence's bucket if multi-sentence).
    public let bucket: ModelBucket?

    /// Audio duration in seconds.
    public var duration: TimeInterval {
        Double(samples.count) / Double(KokoroEngine.sampleRate)
    }

    /// Real-time factor (audio duration / synthesis time).
    /// Values above 1.0 mean faster than real-time.
    public var realTimeFactor: Double {
        synthesisTime > 0 ? duration / synthesisTime : 0
    }

    /// Creates a new synthesis result.
    public init(
        samples: [Float],
        phonemes: String = "",
        tokenDurations: [Int] = [],
        tokenCount: Int,
        synthesisTime: TimeInterval,
        bucket: ModelBucket? = nil
    ) {
        self.samples = samples
        self.phonemes = phonemes
        self.tokenDurations = tokenDurations
        self.tokenCount = tokenCount
        self.synthesisTime = synthesisTime
        self.bucket = bucket
    }
}
