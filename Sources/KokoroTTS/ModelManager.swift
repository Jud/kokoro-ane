import Foundation

/// Validates CoreML model presence for KokoroTTS.
///
/// Models are stored at a fixed path so CoreML preserves its
/// device-specialized compilation cache across launches.
public enum ModelManager {
    /// Suggested model directory for an application.
    ///
    /// Returns `~/Library/Application Support/<bundleIdentifier>/models/kokoro/`.
    ///
    /// - Parameter bundleIdentifier: The app's bundle ID. Defaults to
    ///   `Bundle.main.bundleIdentifier`, falling back to `"kokoro-tts"`.
    public static func defaultDirectory(
        for bundleIdentifier: String = Bundle.main.bundleIdentifier ?? "kokoro-tts"
    ) -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent(bundleIdentifier)
            .appendingPathComponent("models")
            .appendingPathComponent("kokoro")
    }

    /// Check whether enough models exist to run inference.
    ///
    /// Requires at least one frontend+backend `.mlmodelc` pair and a `voices/` directory.
    public static func modelsAvailable(at directory: URL) -> Bool {
        let fm = FileManager.default
        let hasModel = ModelBucket.allCases.contains { bucket in
            fm.fileExists(
                atPath: directory.appendingPathComponent(
                    bucket.frontendModelName + ".mlmodelc"
                ).path)
                && fm.fileExists(
                    atPath: directory.appendingPathComponent(
                        bucket.backendModelName + ".mlmodelc"
                    ).path)
        }
        let hasVoices = fm.fileExists(
            atPath: directory.appendingPathComponent("voices").path)
        return hasModel && hasVoices
    }
}
