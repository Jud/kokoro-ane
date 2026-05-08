import Foundation
import KokoroCoreML
import SwiftUI

/// Deterministic synthesis suite that records phys_footprint at lifecycle
/// events and at 10Hz throughout, then prints a JSON summary and exits.
///
/// Triggered by passing `--memory-test` on launch. Output is fenced with
/// `MEMORY_TEST_RESULT_START` / `MEMORY_TEST_RESULT_END` markers so a CI
/// script can extract the JSON from a noisy log.
enum MemoryTestRunner {
    static var isRequested: Bool {
        CommandLine.arguments.contains("--memory-test")
    }

    @MainActor
    static func runAndExit() async -> Never {
        let recorder = MemoryRecorder()
        recorder.event("startup")

        do {
            let engine = try loadEngine(recorder: recorder)
            try await runSuite(engine: engine, recorder: recorder)
        } catch {
            recorder.event("error", detail: "\(error)")
        }

        recorder.event("teardown")
        let result = recorder.finish()
        emit(result)
        fflush(stdout)
        exit(EXIT_SUCCESS)
    }

    @MainActor
    private static func loadEngine(recorder: MemoryRecorder) throws -> KokoroEngine {
        recorder.event("pre_engine_load")
        let bundledPath = Bundle.main.resourceURL?.appendingPathComponent("Models")
        let modelDir: URL
        if let bundled = bundledPath, KokoroEngine.isDownloaded(at: bundled) {
            modelDir = bundled
        } else {
            modelDir = KokoroEngine.defaultModelDirectory
        }
        let engine = try KokoroEngine(modelDirectory: modelDir)
        recorder.event("post_engine_load", detail: "voices=\(engine.availableVoices.count)")
        return engine
    }

    @MainActor
    private static func runSuite(engine: KokoroEngine, recorder: MemoryRecorder) async throws {
        let voices = pickRepresentativeVoices(from: engine.availableVoices)
        let utterances: [(label: String, text: String)] = [
            ("short", "Hello, how are you today?"),
            (
                "medium",
                String(repeating: "This is a sentence used to exercise medium-length synthesis. ", count: 4)
            ),
            (
                "long",
                String(
                    repeating:
                        "Long-form synthesis exercises the streaming path across many chunks, including inter-chunk silence and the gain locked in from the first chunk. ",
                    count: 6)
            ),
        ]

        for voice in voices {
            for u in utterances {
                try await runOneCase(
                    engine: engine, voice: voice, label: u.label,
                    text: u.text, recorder: recorder)
            }
        }
    }

    @MainActor
    private static func runOneCase(
        engine: KokoroEngine, voice: String, label: String,
        text: String, recorder: MemoryRecorder
    ) async throws {
        recorder.event("case_start", detail: "\(voice)/\(label)")

        let stream = try await Task.detached(priority: .userInitiated) {
            try engine.speak(text, voice: voice, speed: 1.0)
        }.value

        var bufferCount = 0
        for await event in stream {
            switch event {
            case .audio(let buffer):
                bufferCount += 1
                if bufferCount == 1 {
                    recorder.event("first_buffer", detail: "\(voice)/\(label)")
                }
                _ = buffer.frameLength
            case .chunkFailed(let err):
                recorder.event("chunk_failed", detail: err.localizedDescription)
            }
        }

        recorder.event("case_end", detail: "\(voice)/\(label) buffers=\(bufferCount)")
    }

    private static func pickRepresentativeVoices(from voices: [String]) -> [String] {
        let candidates = ["af_heart", "am_michael", "bf_emma"]
        let picked = candidates.filter { voices.contains($0) }
        if !picked.isEmpty { return picked }
        return Array(voices.prefix(3))
    }

    private static func emit(_ result: MemoryTestResult) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard
            let data = try? encoder.encode(result),
            let json = String(data: data, encoding: .utf8)
        else { return }
        print("MEMORY_TEST_RESULT_START")
        print(json)
        print("MEMORY_TEST_RESULT_END")
    }
}

// MARK: - Recorder

@MainActor
final class MemoryRecorder {
    private var events: [MemoryEvent] = []
    private var samples: [MemorySample] = []
    private let baseline: Double
    private let startTime: CFAbsoluteTime
    private var pollTimer: Timer?

    init() {
        startTime = CFAbsoluteTimeGetCurrent()
        baseline = MemoryMeter.physFootprintMB()
        startPolling()
    }

    func event(_ name: String, detail: String? = nil) {
        let t = CFAbsoluteTimeGetCurrent() - startTime
        events.append(
            MemoryEvent(
                name: name, timeSeconds: t,
                mb: MemoryMeter.physFootprintMB(), detail: detail))
    }

    func finish() -> MemoryTestResult {
        pollTimer?.invalidate()
        pollTimer = nil
        let peak = max(
            samples.map(\.mb).max() ?? 0,
            events.map(\.mb).max() ?? 0)
        return MemoryTestResult(
            durationSeconds: CFAbsoluteTimeGetCurrent() - startTime,
            baselineMB: baseline,
            peakMB: peak,
            peakDeltaMB: peak - baseline,
            events: events,
            samples: samples)
    }

    private func startPolling() {
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) {
            [weak self] _ in
            guard let self else { return }
            let t = CFAbsoluteTimeGetCurrent() - self.startTime
            self.samples.append(
                MemorySample(
                    timeSeconds: t, mb: MemoryMeter.physFootprintMB()))
        }
    }
}

// MARK: - Schema

struct MemoryTestResult: Codable {
    let durationSeconds: Double
    let baselineMB: Double
    let peakMB: Double
    let peakDeltaMB: Double
    let events: [MemoryEvent]
    let samples: [MemorySample]
}

struct MemoryEvent: Codable {
    let name: String
    let timeSeconds: Double
    let mb: Double
    let detail: String?
}

struct MemorySample: Codable {
    let timeSeconds: Double
    let mb: Double
}

// MARK: - SwiftUI host

/// Minimal view that shows progress while the test suite runs.
struct MemoryTestView: View {
    @State private var status = "Running memory test..."

    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
            Text(status)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
        .preferredColorScheme(.dark)
        .task {
            await MemoryTestRunner.runAndExit()
        }
    }
}
