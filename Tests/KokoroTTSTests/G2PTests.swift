import Testing

@testable import KokoroTTS

@Suite("G2P")
struct G2PTests {
    private func makeG2P() -> EnglishG2P {
        EnglishG2P(british: false)
    }

    @Test("Simple words produce non-empty phonemes")
    func simpleWords() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "hello world")
        #expect(!phonemes.isEmpty)
        #expect(phonemes != "❓")
    }

    @Test("Numbers are converted to words")
    func numbers() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "42")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("42"))
    }

    @Test("Punctuation preserved")
    func punctuation() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "Hello, world!")
        #expect(phonemes.contains(","))
    }

    @Test("CamelCase OOV splits into known parts")
    func camelCaseSplit() {
        let g2p = makeG2P()
        // "viewDidLoad" — each part (view, did, load) is in the dictionary
        let (phonemes, _) = g2p.phonemize(text: "viewDidLoad")
        #expect(!phonemes.contains("❓"))
    }

    @Test("Acronyms spelled out")
    func acronymSpelling() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "API")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("❓"))
    }

    @Test("Empty text returns empty phonemes")
    func emptyText() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "")
        #expect(phonemes.isEmpty)
    }

    @Test("Mixed text with code terms")
    func mixedText() {
        let g2p = makeG2P()
        let (phonemes, _) = g2p.phonemize(text: "The UIViewController handles user input.")
        #expect(!phonemes.isEmpty)
        #expect(!phonemes.contains("❓"))
    }
}
