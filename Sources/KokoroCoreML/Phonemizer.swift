/// Protocol for converting text to IPA phoneme strings.
///
/// KokoroCoreML ships with BART G2P as the default phonemizer.
/// Implement this protocol to substitute your own phonemization backend
/// (e.g., eSpeak-NG for multilingual support).
public protocol Phonemizer: Sendable {
    /// Convert text to an IPA phoneme string suitable for Kokoro tokenization.
    func phonemize(_ text: String) -> String
}

extension EnglishG2P: @unchecked Sendable {}

extension EnglishG2P: Phonemizer {
    func phonemize(_ text: String) -> String {
        let (phonemes, _) = self.phonemize(text: text)
        return phonemes
    }
}
