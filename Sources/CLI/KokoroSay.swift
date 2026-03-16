import ArgumentParser
import AVFoundation
import Foundation
import KokoroTTS

extension ModelBucket: ExpressibleByArgument {}

@main
struct KokoroSay: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kokoro-say",
        abstract: "Text-to-speech using KokoroTTS",
        version: "1.0.0"
    )

    @Option(name: [.short, .long], help: "Voice preset")
    var voice: String = "af_heart"

    @Option(name: [.short, .long], help: "Speed multiplier, 0.5–2.0")
    var speed: Float = 1.0

    @Option(name: [.short, .long], help: "Write WAV to file")
    var output: String?

    @Option(name: .long, help: "Model directory path")
    var modelDir: String?

    @Option(name: .long, help: "Force model bucket")
    var bucket: ModelBucket?

    @Flag(name: [.short, .long], help: "Play audio through speakers")
    var play = false

    @Flag(name: .long, help: "Skip audio post-processing")
    var raw = false

    @Flag(name: .long, help: "Stream audio (start playback before full synthesis)")
    var stream = false

    @Flag(name: .long, help: "Print debug information")
    var debug = false

    @Flag(name: .long, help: "List available voices")
    var listVoices = false

    @Argument(help: "Text to synthesize (reads stdin if omitted)")
    var text: [String] = []

    func validate() throws {
        guard KokoroEngine.speedRange.contains(speed) else {
            throw ValidationError("Speed must be between \(KokoroEngine.speedRange.lowerBound) and \(KokoroEngine.speedRange.upperBound)")
        }
    }

    mutating func run() async throws {
        let dir: URL
        if let modelDir {
            dir = URL(fileURLWithPath: modelDir)
        } else {
            dir = ModelManager.defaultDirectory()
        }

        if !ModelManager.modelsAvailable(at: dir) {
            try ModelDownloader.download(to: dir)
            guard ModelManager.modelsAvailable(at: dir) else {
                fputs("Download completed but models could not be loaded.\n", stderr)
                throw ExitCode.failure
            }
        }

        let engine = try KokoroEngine(modelDirectory: dir)

        if listVoices {
            for v in engine.availableVoices.sorted() { print(v) }
            return
        }

        let inputText = try resolveText()

        guard engine.availableVoices.contains(voice) else {
            fputs("Unknown voice '\(voice)'. Available:\n", stderr)
            for v in engine.availableVoices.sorted() { fputs("  \(v)\n", stderr) }
            throw ExitCode.failure
        }

        if debug {
            print("Model dir: \(dir.path)")
            print("Loaded buckets: \(engine.activeBuckets.map(\.modelName).joined(separator: ", "))")
        }

        if stream && output == nil {
            try await streamPlayback(engine: engine, text: inputText)
        } else {
            let result = try engine.synthesize(
                text: inputText, voice: voice, speed: speed,
                bucket: bucket, rawAudio: raw
            )

            if debug { printDebugInfo(result: result) }

            let tag = result.bucket.map { " \($0.modelName)" } ?? ""
            let stats = String(
                format: "%.0fms synth, %.1fs audio, %.1fx RT",
                result.synthesisTime * 1000, result.duration, result.realTimeFactor
            )
            print("[\(voice)\(tag)] \(stats)")

            if let output {
                try writeWAV(samples: result.samples, to: output)
                print("Wrote \(output)")
            }

            if play || (output == nil && !debug) {
                try playAudio(samples: result.samples)
            }
        }
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
            AVSampleRateKey: Double(KokoroEngine.sampleRate),
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

        for await buffer in try engine.speak(text, voice: voice, speed: speed) {
            chunks += 1
            totalFrames += buffer.frameLength
            player.scheduleBuffer(buffer, completionHandler: nil)
            if !reportedFirst {
                reportedFirst = true
                let latency = CFAbsoluteTimeGetCurrent() - t0
                print("[\(voice)] first audio in \(Int(latency * 1000))ms")
            }
        }

        let duration = Double(totalFrames) / Double(KokoroEngine.sampleRate)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("[\(voice)] \(chunks) chunks, \(String(format: "%.1f", duration))s audio, \(Int(elapsed * 1000))ms total synth")

        let sentinel = AVAudioPCMBuffer(pcmFormat: KokoroEngine.audioFormat, frameCapacity: 1)!
        sentinel.frameLength = 1
        sentinel.floatChannelData?[0].pointee = 0
        await player.scheduleBuffer(sentinel)
        try await Task.sleep(for: .milliseconds(100))
    }

    // MARK: - Debug

    private func printDebugInfo(result: SynthesisResult) {
        print("Phonemes: \(result.phonemes)")
        if let bucket = result.bucket {
            print("Selected bucket: \(bucket.modelName) (\(bucket.maxTokens) max tokens)")
        }
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
        print(String(format: "  Token durations (%d): %@", result.tokenDurations.count,
            result.tokenDurations.map(String.init).joined(separator: ", ")))

        let hopSize = KokoroEngine.hopSize
        var cumFrames = 0
        for (i, dur) in result.tokenDurations.enumerated() {
            let startSample = cumFrames * hopSize
            let endSample = min((cumFrames + dur) * hopSize, result.samples.count)
            var peak: Float = 0
            if startSample < endSample {
                for j in startSample..<endSample { peak = max(peak, abs(result.samples[j])) }
            }
            print(String(format: "    t%02d: dur=%2d  %6d-%6d  peak=%.3f",
                i, dur, startSample, endSample, peak))
            cumFrames += dur
        }
    }
}
