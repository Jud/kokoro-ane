import Foundation

enum ModelDownloader {
    static let repo = "Jud/kokoro-tts-swift"
    static let asset = "kokoro-models.tar.gz"

    static func latestModelTag() throws -> String {
        let url = URL(string: "https://api.github.com/repos/\(repo)/releases")!
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<String, Error> = .failure(URLError(.unknown))

        let task = URLSession.shared.dataTask(with: url) { data, _, error in
            if let error {
                result = .failure(error)
            } else if let data,
                let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
            {
                if let release = json.first(where: {
                    ($0["tag_name"] as? String)?.hasPrefix("models-") == true
                }),
                    let tag = release["tag_name"] as? String
                {
                    result = .success(tag)
                } else {
                    result = .failure(URLError(.resourceUnavailable))
                }
            } else {
                result = .failure(URLError(.cannotParseResponse))
            }
            sem.signal()
        }
        task.resume()
        sem.wait()
        return try result.get()
    }

    static func assetURL(tag: String) -> URL {
        URL(string: "https://github.com/\(repo)/releases/download/\(tag)/\(asset)")!
    }

    private static func installedTag(at directory: URL) -> String? {
        let tagFile = directory.appendingPathComponent(".model-tag")
        return try? String(contentsOf: tagFile, encoding: .utf8).trimmingCharacters(
            in: .whitespacesAndNewlines)
    }

    private static func writeInstalledTag(_ tag: String, at directory: URL) {
        let tagFile = directory.appendingPathComponent(".model-tag")
        try? tag.write(to: tagFile, atomically: true, encoding: .utf8)
    }

    static func isUpToDate(at directory: URL) -> Bool {
        guard let installed = installedTag(at: directory) else { return false }
        guard let latest = try? latestModelTag() else { return true }
        return installed == latest
    }

    static func download(
        to directory: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)

        let tag: String
        do {
            tag = try latestModelTag()
        } catch {
            tag = "models-v1"
        }
        let url = assetURL(tag: tag)

        let tmpDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpDir) }

        let tarball = tmpDir.appendingPathComponent(asset)
        let tarballPath = tarball.path

        // Download
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var downloadResult: Result<Void, Error> = .success(())

        let task = URLSession.shared.downloadTask(with: url) { url, response, error in
            if let error {
                downloadResult = .failure(error)
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                downloadResult = .failure(URLError(.badServerResponse))
            } else if let url {
                do {
                    try FileManager.default.moveItem(
                        at: url, to: URL(fileURLWithPath: tarballPath))
                } catch {
                    downloadResult = .failure(error)
                }
            }
            sem.signal()
        }

        let observation: NSKeyValueObservation?
        if let progress {
            nonisolated(unsafe) var lastPct = -1
            observation = task.progress.observe(\.fractionCompleted) { prog, _ in
                let pct = Int(prog.fractionCompleted * 100)
                guard pct != lastPct else { return }
                lastPct = pct
                progress(prog.fractionCompleted)
            }
        } else {
            observation = nil
        }

        task.resume()
        sem.wait()
        observation?.invalidate()

        try downloadResult.get()

        // Extract
        #if os(macOS)
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            proc.arguments = ["xzf", tarball.path, "-C", directory.path]
            try proc.run()
            proc.waitUntilExit()
            guard proc.terminationStatus == 0 else {
                throw CocoaError(.fileReadCorruptFile)
            }
        #else
            throw KokoroError.inferenceFailed(
                "Automatic model download is only supported on macOS. "
                    + "On iOS, bundle models with your app or download and extract manually.")
        #endif

        writeInstalledTag(tag, at: directory)
    }
}
