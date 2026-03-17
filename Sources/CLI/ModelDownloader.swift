import ArgumentParser
import Foundation
import KokoroTTS

enum CLIModelDownloader {
    static func downloadWithProgress(to directory: URL) throws {
        fputs("Downloading KokoroTTS models...\n", stderr)
        let barWidth = 30
        nonisolated(unsafe) var lastPct = -1

        try KokoroEngine.download(to: directory) { fraction in
            let pct = Int(fraction * 100)
            guard pct != lastPct else { return }
            lastPct = pct
            let filled = Int(fraction * Double(barWidth))
            let bar =
                String(repeating: "█", count: filled)
                + String(repeating: "░", count: barWidth - filled)
            fputs("\r  \(bar) \(pct)%", stderr)
        }
        fputs("\n  Models installed to \(directory.path)\n", stderr)
    }

    static func ensureModels(at directory: URL) throws {
        if !KokoroEngine.isDownloaded(at: directory) {
            try downloadWithProgress(to: directory)
            guard KokoroEngine.isDownloaded(at: directory) else {
                fputs("Download completed but models could not be loaded.\n", stderr)
                throw ExitCode.failure
            }
        }
    }
}
