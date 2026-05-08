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
        isSynthesizing = true
        error = nil

        streamTask = Task { [weak self] in
            await self?.run(engine: engine, text: text, voice: voice, speed: speed)
        }
    }

    func stop() {
        streamTask?.cancel()
        streamTask = nil
        teardown()
        isSynthesizing = false
    }

    // MARK: - Private

    private func run(engine: KokoroEngine, text: String, voice: String, speed: Float) async {
        let player: AVAudioPlayerNode
        do {
            player = try startPlayback()
        } catch {
            self.error = "Playback setup failed: \(error.localizedDescription)"
            self.isSynthesizing = false
            return
        }

        do {
            let stream = try engine.speak(text, voice: voice, speed: speed)
            var firstBuffer = true
            for await event in stream {
                if Task.isCancelled { break }
                switch event {
                case .audio(let buffer):
                    if firstBuffer {
                        firstBuffer = false
                        self.isPlaying = true
                    }
                    player.scheduleBuffer(buffer, completionHandler: nil)
                case .chunkFailed(let err):
                    self.error = "Chunk failed: \(err.localizedDescription)"
                }
            }
        } catch {
            self.error = "Synthesis failed: \(error.localizedDescription)"
            teardown()
            return
        }

        self.isSynthesizing = false

        if Task.isCancelled {
            teardown()
            return
        }

        scheduleEndSentinel(on: player)
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
        let tapFormat = avEngine.mainMixerNode.outputFormat(forBus: 0)
        avEngine.mainMixerNode.installTap(
            onBus: 0, bufferSize: 1024, format: tapFormat
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

    private func scheduleEndSentinel(on player: AVAudioPlayerNode) {
        guard
            let sentinel = AVAudioPCMBuffer(
                pcmFormat: KokoroEngine.audioFormat, frameCapacity: 1)
        else {
            teardown()
            return
        }
        sentinel.frameLength = 1
        sentinel.floatChannelData?[0].pointee = 0
        player.scheduleBuffer(
            sentinel, at: nil, options: [], completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            Task { @MainActor [weak self] in self?.teardown() }
        }
    }

    private func teardown() {
        playerNode?.stop()
        avEngine?.mainMixerNode.removeTap(onBus: 0)
        avEngine?.stop()
        avEngine = nil
        playerNode = nil
        isPlaying = false
        levelMonitor?.reset()
    }
}
