#if ESPEAK_NG
    import Foundation
    import libespeak_ng

    /// Phonemizer using eSpeak-NG for multilingual IPA output.
    ///
    /// Requires the `espeak` SPM trait to be enabled:
    /// ```
    /// swift build --traits espeak
    /// ```
    ///
    /// - Important: eSpeak-NG is GPL-3.0 licensed. Enabling this trait
    ///   makes your binary subject to GPL-3.0 terms.
    public final class EspeakPhonemizer: Phonemizer, @unchecked Sendable {

        private let language: String
        private let lock = NSLock()

        /// Create an eSpeak-NG phonemizer.
        ///
        /// - Parameter language: eSpeak voice name (e.g. "en", "fr", "ja").
        ///   Defaults to "en" (English).
        /// - Throws: If eSpeak-NG initialization fails.
        public init(language: String = "en") throws {
            self.language = language

            // Install compiled espeak data (phonemes, dictionaries) on first run.
            // The SPM bundle ships source data that must be compiled once at runtime.
            let root = try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask,
                appropriateFor: nil, create: true
            )
            .appendingPathComponent("kokoro-espeak")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

            try EspeakLib.ensureBundleInstalled(inRoot: root)

            espeak_ng_InitializePath(root.path)

            // Suppress "Can't read dictionary file" warnings from eSpeak init.
            // These are benign — eSpeak scans for all ~120 language dictionaries
            // but most aren't compiled from source data. The languages we need
            // (fr, es, it, pt, hi) are compiled by ensureBundleInstalled above.
            let savedStderr = dup(STDERR_FILENO)
            let devNull = open("/dev/null", O_WRONLY)
            dup2(devNull, STDERR_FILENO)
            let status = espeak_ng_Initialize(nil)
            dup2(savedStderr, STDERR_FILENO)
            close(devNull)
            close(savedStderr)

            guard status == ENS_OK else {
                throw KokoroError.modelLoadFailed(
                    "espeak_ng_Initialize failed with status \(status.rawValue)")
            }

            let voiceStatus = espeak_ng_SetVoiceByName(language)
            guard voiceStatus == ENS_OK else {
                throw KokoroError.modelLoadFailed(
                    "espeak_ng_SetVoiceByName('\(language)') failed with status \(voiceStatus.rawValue)")
            }

            espeak_ng_SetPhonemeEvents(1, 0)
        }

        deinit {
            espeak_Terminate()
        }

        public func phonemize(_ text: String) -> String {
            lock.lock()
            defer { lock.unlock() }

            return text.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
                .map { phonemizeLine($0) }
                .joined(separator: " ")
        }

        private func phonemizeLine(_ line: String) -> String {
            let textMode: Int32 = 1  // espeakCHARS_UTF8
            let phonemeMode: Int32 = 0x02  // espeakPHONEMES_IPA

            var result = ""
            line.withCString { cString in
                var ptr: UnsafePointer<CChar>? = cString
                withUnsafeMutablePointer(to: &ptr) { mutablePtr in
                    let rawPtr = mutablePtr.withMemoryRebound(
                        to: UnsafeRawPointer?.self, capacity: 1
                    ) { $0 }
                    if let phonemes = espeak_TextToPhonemes(rawPtr, textMode, phonemeMode) {
                        result = String(cString: phonemes)
                    }
                }
            }
            return result.trimmingCharacters(in: .whitespaces)
        }
    }
#endif
