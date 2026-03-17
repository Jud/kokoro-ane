import Foundation
import Testing

@testable import KokoroTTS

@Suite(
    "Inference Benchmark",
    .enabled(if: ModelManager.modelsAvailable(at: ModelManager.defaultDirectory())),
    .serialized
)
struct InferenceBenchmarkTests {
    static let runs = 5

    // swiftlint:disable:next force_try
    static let engine = try! KokoroEngine(modelDirectory: ModelManager.defaultDirectory())

    @Test("Short sentence — single chunk synthesis")
    func shortSentence() throws {
        let text = "Hello, welcome to Kokoro text to speech."
        let stats = try measure(text: text)
        print("Short: \(stats)")
        #expect(stats.medianRTF > 1.0, "Median slower than real-time: \(fmt(stats.medianRTF))x")
    }

    @Test("Medium paragraph — multi-chunk synthesis")
    func mediumParagraph() throws {
        let text = """
            The quick brown fox jumps over the lazy dog near the bank of the river. \
            This is a longer piece of text that should be split into multiple chunks \
            by the synthesis engine automatically.
            """
        let stats = try measure(text: text)
        print("Medium: \(stats)")
        #expect(stats.medianRTF > 1.0, "Median slower than real-time: \(fmt(stats.medianRTF))x")
    }

    @Test("Long multi-paragraph — sustained synthesis")
    func longText() throws {
        let text = """
            Artificial intelligence has transformed the way we interact with technology. \
            From voice assistants to autonomous vehicles, AI systems are becoming an integral \
            part of daily life.

            Text to speech technology, in particular, has seen remarkable advances. Modern \
            neural systems can produce speech that is nearly indistinguishable from human \
            voices. The Kokoro model represents a lightweight approach that runs efficiently \
            on edge devices using the Apple Neural Engine.
            """
        let stats = try measure(text: text)
        print("Long: \(stats)")
        #expect(stats.medianRTF > 1.0, "Median slower than real-time: \(fmt(stats.medianRTF))x")
    }

    // MARK: - Helpers

    private struct Stats: CustomStringConvertible {
        let medianRTF: Double
        let minRTF: Double
        let maxRTF: Double
        let medianSynthMs: Double
        let audioDuration: Double
        let runs: Int

        var description: String {
            "median \(fmt(medianRTF))x RTF (\(fmt(minRTF))x–\(fmt(maxRTF))x), "
                + "\(fmt(medianSynthMs))ms synth, \(fmt(audioDuration))s audio, \(runs) runs"
        }
    }

    private func measure(text: String) throws -> Stats {
        let audioDuration = try Self.engine.synthesize(text: text, voice: "af_heart").duration
        var rtfs: [Double] = []
        var synthTimes: [Double] = []

        for _ in 0..<Self.runs {
            let result = try Self.engine.synthesize(text: text, voice: "af_heart")
            rtfs.append(result.realTimeFactor)
            synthTimes.append(result.synthesisTime)
        }

        rtfs.sort()
        synthTimes.sort()
        let mid = Self.runs / 2

        return Stats(
            medianRTF: rtfs[mid],
            minRTF: rtfs[0],
            maxRTF: rtfs[Self.runs - 1],
            medianSynthMs: synthTimes[mid] * 1000,
            audioDuration: audioDuration,
            runs: Self.runs
        )
    }
}

private func fmt(_ value: Double) -> String {
    String(format: "%.1f", value)
}
