import AVFoundation
import KokoroCoreML
import Observation

@Observable
@MainActor
final class AudioEngine {
    var isPlaying = false
    var isSynthesizing = false
    var availableVoices: [String] = []
    var isReady = false
    var error: String?
    var levelMonitor: AudioLevelMonitor?

    private var engine: KokoroEngine?
    private var avEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamTask: Task<Void, Never>?
    private var generation: UInt64 = 0
    private var queuedFrames: AVAudioFramePosition = 0
    private var drainContinuation: CheckedContinuation<Void, Never>?
    private static let maxQueuedSeconds: Double = 2.0
    private var maxQueuedFrames: AVAudioFramePosition {
        AVAudioFramePosition(Self.maxQueuedSeconds * Double(KokoroEngine.sampleRate))
    }
    private var _analyzer: SpectrumAnalyzer?
    private var spectrumAnalyzer: SpectrumAnalyzer {
        if let a = _analyzer { return a }
        let a = SpectrumAnalyzer(frameCount: 1024, sampleRate: Float(KokoroEngine.sampleRate))
        _analyzer = a
        return a
    }

    func load(modelDirectory: URL) {
        do {
            let kokoroEngine = try KokoroEngine(modelDirectory: modelDirectory)
            self.engine = kokoroEngine
            self.availableVoices = kokoroEngine.availableVoices
            self.isReady = true
        } catch {
            self.error = "Failed to load models: \(error.localizedDescription)"
        }
    }

    func speak(text: String, voice: String, speed: Float = 1.0) {
        guard let engine, !text.isEmpty else { return }

        stop()
        let myGeneration = generation
        isSynthesizing = true
        error = nil

        streamTask = Task { [weak self] in
            await self?.run(
                engine: engine, text: text, voice: voice,
                speed: speed, generation: myGeneration)
        }
    }

    func stop() {
        generation &+= 1
        streamTask?.cancel()
        streamTask = nil
        teardown()
        isSynthesizing = false
    }

    // MARK: - Private

    private func run(
        engine: KokoroEngine, text: String, voice: String,
        speed: Float, generation myGeneration: UInt64
    ) async {
        func current() -> Bool {
            !Task.isCancelled && self.generation == myGeneration
        }

        let player: AVAudioPlayerNode
        do {
            player = try startPlayback()
        } catch {
            if current() {
                self.error = "Playback setup failed: \(error.localizedDescription)"
                self.isSynthesizing = false
            }
            return
        }

        let stream: AsyncStream<SpeakEvent>
        do {
            stream = try await Task.detached(priority: .userInitiated) {
                try engine.speak(text, voice: voice, speed: speed)
            }.value
        } catch {
            if current() {
                self.error = "Synthesis failed: \(error.localizedDescription)"
                self.isSynthesizing = false
                teardown()
            }
            return
        }

        var firstBuffer = true
        for await event in stream {
            if !current() { break }
            switch event {
            case .audio(let buffer):
                if firstBuffer {
                    firstBuffer = false
                    self.isSynthesizing = false
                    self.isPlaying = true
                }
                await enqueue(buffer: buffer, on: player, generation: myGeneration)
            case .chunkFailed(let err):
                self.error = "Chunk failed: \(err.localizedDescription)"
            }
        }

        if !current() { return }

        self.isSynthesizing = false
        scheduleEndSentinel(on: player, generation: myGeneration)
    }

    private func startPlayback() throws -> AVAudioPlayerNode {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)

        let avEngine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        avEngine.attach(player)
        avEngine.connect(
            player, to: avEngine.mainMixerNode, format: KokoroEngine.audioFormat)

        let analyzer = self.spectrumAnalyzer
        let monitor = self.levelMonitor
        player.installTap(
            onBus: 0, bufferSize: 1024, format: KokoroEngine.audioFormat
        ) { buffer, _ in
            let bands = analyzer.analyze(buffer)
            monitor?.pushBands(bands)
        }

        avEngine.prepare()
        try avEngine.start()
        player.play()

        self.avEngine = avEngine
        self.playerNode = player
        return player
    }

    private func enqueue(
        buffer: AVAudioPCMBuffer, on player: AVAudioPlayerNode, generation myGeneration: UInt64
    ) async {
        let frames = AVAudioFramePosition(buffer.frameLength)
        while self.generation == myGeneration, !Task.isCancelled,
            queuedFrames > maxQueuedFrames
        {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                drainContinuation = cont
            }
        }
        guard self.generation == myGeneration, !Task.isCancelled else { return }
        queuedFrames += frames
        player.scheduleBuffer(
            buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == myGeneration else { return }
                self.queuedFrames -= frames
                if self.queuedFrames <= self.maxQueuedFrames {
                    let cont = self.drainContinuation
                    self.drainContinuation = nil
                    cont?.resume()
                }
            }
        }
    }

    private func scheduleEndSentinel(on player: AVAudioPlayerNode, generation myGeneration: UInt64) {
        guard
            let sentinel = AVAudioPCMBuffer(
                pcmFormat: KokoroEngine.audioFormat, frameCapacity: 1)
        else {
            if self.generation == myGeneration { teardown() }
            return
        }
        sentinel.frameLength = 1
        sentinel.floatChannelData?[0].pointee = 0
        player.scheduleBuffer(
            sentinel, at: nil, options: [], completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == myGeneration else { return }
                self.teardown()
            }
        }
    }

    private func teardown() {
        let pending = drainContinuation
        drainContinuation = nil
        pending?.resume()
        queuedFrames = 0
        playerNode?.removeTap(onBus: 0)
        playerNode?.stop()
        avEngine?.stop()
        avEngine = nil
        playerNode = nil
        isPlaying = false
        levelMonitor?.reset()
    }
}
