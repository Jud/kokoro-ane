import Foundation
import Testing

@testable import KokoroTTS

@Suite("Num2Word")
struct Num2WordTests {
    @Test("Year 2025")
    func year2025() {
        let n2w = EnglishNum2Word()
        let result = n2w.convert(Decimal(2025), to: .year)
        #expect(result == "twenty twenty-five")
    }

    @Test("Cardinal small numbers")
    func cardinalSmall() {
        let n2w = EnglishNum2Word()
        #expect(n2w.convert(Decimal(0)) == "zero")
        #expect(n2w.convert(Decimal(1)) == "one")
        #expect(n2w.convert(Decimal(13)) == "thirteen")
        #expect(n2w.convert(Decimal(20)) == "twenty")
        #expect(n2w.convert(Decimal(21)) == "twenty-one")
        #expect(n2w.convert(Decimal(98)) == "ninety-eight")
        #expect(n2w.convert(Decimal(100)) == "one hundred")
    }

    @Test("Cardinal large numbers")
    func cardinalLarge() {
        let n2w = EnglishNum2Word()
        #expect(n2w.convert(Decimal(2025)) == "two thousand, twenty-five")
        #expect(n2w.convert(Decimal(1000)) == "one thousand")
        #expect(n2w.convert(Decimal(42)) == "forty-two")
    }

    @Test("Ordinals")
    func ordinals() {
        let n2w = EnglishNum2Word()
        #expect(n2w.convert(Decimal(1), to: .ordinal) == "first")
        #expect(n2w.convert(Decimal(3), to: .ordinal) == "third")
        #expect(n2w.convert(Decimal(21), to: .ordinal) == "twenty-first")
    }
}
