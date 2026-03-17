import Foundation
import Testing

@testable import KokoroTTS

@Suite("BART G2P Integration")
struct BARTG2PIntegrationTests {
    @Test("OOV technical terms produce valid phonemes via G2P pipeline")
    func oovThroughG2P() {
        let g2p = EnglishG2P(british: false)
        let words = ["kubernetes", "nginx", "graphql", "anthropic"]
        for word in words {
            let (phonemes, _) = g2p.phonemize(text: word)
            #expect(!phonemes.isEmpty, "\(word) produced empty phonemes")
            #expect(!phonemes.contains("❓"), "\(word) hit unknown fallback")
        }
    }
}
