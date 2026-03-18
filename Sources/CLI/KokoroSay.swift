import AVFoundation
import ArgumentParser
import Foundation
import KokoroTTS

@main
struct Kokoro: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kokoro",
        abstract: "Kokoro text-to-speech",
        version: "0.3.0",
        subcommands: [Say.self, Update.self, Daemon.self],
        defaultSubcommand: Say.self
    )
}

// MARK: - Say

struct Say: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Synthesize text to speech"
    )

    @Option(name: [.short, .long], help: "Voice preset")
    var voice: String = "af_heart"

    @Option(name: [.short, .long], help: "Speed multiplier, 0.5–2.0")
    var speed: Float = 1.0

    @Option(name: [.short, .long], help: "Write WAV to file")
    var output: String?

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    @Flag(name: [.short, .long], help: "Play audio through speakers")
    var play = false

    @Flag(name: .long, help: "Stream audio (start playback before full synthesis)")
    var stream = false

    @Flag(name: .long, help: "Input is IPA phonemes (skip G2P)")
    var ipa = false

    @Flag(name: .long, help: "Print debug information")
    var debug = false

    @Flag(name: .long, help: "List available voices")
    var listVoices = false

    @Argument(help: "Text to synthesize (reads stdin if omitted)")
    var text: [String] = []

    func validate() throws {
        guard (0.5...2.0).contains(speed) else {
            throw ValidationError("Speed must be between 0.5 and 2.0")
        }
        if stream && ipa {
            throw ValidationError("--stream and --ipa cannot be used together yet")
        }
    }

    mutating func run() async throws {
        if stream && output == nil {
            // Streaming needs async for the AVAudioPlayerNode await.
            // speak() runs CoreML inference on its own internal Task.
            try await executeStreaming()
        } else {
            // Run on a regular thread — CoreML inference overflows the
            // cooperative thread pool's small stacks.
            let say = self
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                let thread = Thread {
                    do {
                        try say.execute()
                        cont.resume()
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
                thread.stackSize = 8 * 1024 * 1024  // 8MB stack for CoreML
                thread.start()
            }
        }
    }

    private func execute() throws {
        // --list-voices needs the engine, handle separately
        if listVoices {
            let engine = try loadEngine()
            for v in engine.availableVoices.sorted() { print(v) }
            return
        }

        // Resolve text once for both paths
        let inputText = try resolveText()

        // Try daemon (unless --debug or --ipa which need local engine)
        if !debug && !ipa {
            let request = SynthesisRequest(
                text: inputText, voice: voice, speed: speed)
            switch DaemonClient.synthesize(request) {
            case .success(let response, let samples):
                let duration = Double(samples.count) / KokoroEngine.audioFormat.sampleRate
                let synthMs = (response.synthesisTime ?? 0) * 1000
                let rt =
                    (response.synthesisTime ?? 0) > 0
                    ? duration / response.synthesisTime! : 0
                let stats = String(
                    format: "%.0fms synth, %.1fs audio, %.1fx RT", synthMs, duration, rt)
                print("[\(voice) daemon] \(stats)")
                if let output {
                    try writeWAV(samples: samples, to: output)
                    print("Wrote \(output)")
                }
                if play || output == nil {
                    try playAudio(samples: samples)
                }
                return
            case .daemonError(let message):
                fputs("Daemon error: \(message)\n", stderr)
                throw ExitCode.failure
            case .unavailable:
                break  // fall through to direct engine
            }
        }

        // Direct engine path
        let engine = try loadEngine()

        guard engine.availableVoices.contains(voice) else {
            fputs("Unknown voice '\(voice)'. Available:\n", stderr)
            for v in engine.availableVoices.sorted() { fputs("  \(v)\n", stderr) }
            throw ExitCode.failure
        }

        if debug {
            let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
            print("Model dir: \(dir.path)")
        }

        let result: SynthesisResult
        if ipa {
            result = try engine.synthesize(
                ipa: inputText, voice: voice, speed: speed)
        } else {
            result = try engine.synthesize(
                text: inputText, voice: voice, speed: speed)
        }

        if debug { printDebugInfo(result: result) }

        let stats = String(
            format: "%.0fms synth, %.1fs audio, %.1fx RT",
            result.synthesisTime * 1000, result.duration, result.realTimeFactor
        )
        print("[\(voice)] \(stats)")
        if let output {
            try writeWAV(samples: result.samples, to: output)
            print("Wrote \(output)")
        }
        if play || (output == nil && !debug) {
            try playAudio(samples: result.samples)
        }

        // Occasional daemon hint (not in --debug mode where user chose local)
        if !debug && Int.random(in: 0..<3) == 0 {
            fputs("Tip: run 'kokoro daemon start' for faster synthesis\n", stderr)
        }
    }

    private func loadEngine() throws -> KokoroEngine {
        let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
        try CLIModelDownloader.ensureModels(at: dir)
        return try KokoroEngine(modelDirectory: dir)
    }

    private func executeStreaming() async throws {
        let engine = try loadEngine()
        let inputText = try resolveText()

        guard engine.availableVoices.contains(voice) else {
            fputs("Unknown voice '\(voice)'. Available:\n", stderr)
            for v in engine.availableVoices.sorted() { fputs("  \(v)\n", stderr) }
            throw ExitCode.failure
        }

        try await streamPlayback(engine: engine, text: inputText)
    }

    // MARK: - Input

    private func resolveText() throws -> String {
        if !text.isEmpty {
            return text.joined(separator: " ")
        }
        guard isatty(fileno(stdin)) == 0 else {
            fputs("No text provided. Pass text as arguments or pipe to stdin.\n", stderr)
            throw ExitCode.failure
        }
        var lines: [String] = []
        while let line = readLine() {
            lines.append(line)
        }
        let result = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !result.isEmpty else {
            fputs("Empty input from stdin\n", stderr)
            throw ExitCode.failure
        }
        return result
    }

    // MARK: - Audio Helpers

    private func startAudioPlayer() throws -> (AVAudioEngine, AVAudioPlayerNode) {
        let audioEngine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        audioEngine.attach(player)
        audioEngine.connect(player, to: audioEngine.mainMixerNode, format: KokoroEngine.audioFormat)
        try audioEngine.start()
        player.play()
        return (audioEngine, player)
    }

    private func writeWAV(samples: [Float], to path: String) throws {
        let url = URL(fileURLWithPath: path)
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: KokoroEngine.audioFormat.sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        guard let buf = KokoroEngine.makePCMBuffer(from: samples, format: file.processingFormat)
        else {
            fputs("Failed to create audio buffer\n", stderr)
            throw ExitCode.failure
        }
        try file.write(from: buf)
    }

    private func playAudio(samples: [Float]) throws {
        let (audioEngine, player) = try startAudioPlayer()
        defer { audioEngine.stop() }

        guard let buf = KokoroEngine.makePCMBuffer(from: samples, format: KokoroEngine.audioFormat)
        else {
            fputs("Failed to create audio buffer\n", stderr)
            throw ExitCode.failure
        }
        let done = DispatchSemaphore(value: 0)
        player.scheduleBuffer(buf) { done.signal() }
        done.wait()
        Thread.sleep(forTimeInterval: 0.1)
    }

    // MARK: - Streaming

    private func streamPlayback(engine: KokoroEngine, text: String) async throws {
        let (audioEngine, player) = try startAudioPlayer()
        defer { audioEngine.stop() }

        let t0 = CFAbsoluteTimeGetCurrent()
        var chunks = 0
        var totalFrames: AVAudioFrameCount = 0
        var reportedFirst = false

        for await event in try engine.speak(text, voice: voice, speed: speed) {
            switch event {
            case .audio(let buffer):
                chunks += 1
                totalFrames += buffer.frameLength
                player.scheduleBuffer(buffer, completionHandler: nil)
                if !reportedFirst {
                    reportedFirst = true
                    let latency = CFAbsoluteTimeGetCurrent() - t0
                    print("[\(voice)] first audio in \(Int(latency * 1000))ms")
                }
            case .chunkFailed(let error):
                print("[\(voice)] chunk failed: \(error.localizedDescription)")
            }
        }

        let duration = Double(totalFrames) / KokoroEngine.audioFormat.sampleRate
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let synthMs = Int(elapsed * 1000)
        let durStr = String(format: "%.1f", duration)
        print("[\(voice)] \(chunks) chunks, \(durStr)s audio, \(synthMs)ms total synth")

        let sentinel = AVAudioPCMBuffer(pcmFormat: KokoroEngine.audioFormat, frameCapacity: 1)!
        sentinel.frameLength = 1
        sentinel.floatChannelData?[0].pointee = 0
        await player.scheduleBuffer(sentinel)
        try await Task.sleep(for: .milliseconds(100))
    }

    // MARK: - Debug

    private func printDebugInfo(result: SynthesisResult) {
        print("Phonemes: \(result.phonemes)")
        let windowSize = 120
        let windows = min(20, result.samples.count / windowSize)
        print("\nOnset amplitude profile (first \(windows * 5)ms):")
        for w in 0..<windows {
            let start = w * windowSize
            let end = min(start + windowSize, result.samples.count)
            var peak: Float = 0
            for i in start..<end { peak = max(peak, abs(result.samples[i])) }
            let bar = String(repeating: "#", count: Int(peak * 50))
            print(String(format: "  %3d-%3dms: %.3f %@", w * 5, (w + 1) * 5, peak, bar))
        }
        var globalPeak: Float = 0
        for s in result.samples { globalPeak = max(globalPeak, abs(s)) }
        print(String(format: "\n  Global peak: %.3f", globalPeak))
        print(String(format: "  Total samples: %d (%.1fs)", result.samples.count, result.duration))
    }
}

// MARK: - Update

struct Update: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download latest models"
    )

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    mutating func run() async throws {
        let dir = modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
        try CLIModelDownloader.downloadWithProgress(to: dir)
        guard KokoroEngine.isDownloaded(at: dir) else {
            fputs("Download completed but models could not be loaded.\n", stderr)
            throw ExitCode.failure
        }
    }
}
