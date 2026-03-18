// swift-format-ignore-file
// Originally from MisakiSwift by mlalma, Apache License 2.0
// BART neural fallback replaced with CamelCase splitting + letter spelling.

import BARTG2P
import Foundation
import NaturalLanguage

final class EnglishG2P {
    private let british: Bool
    private let tagger: NLTagger
    private let lexicon: Lexicon
    private let unk: String
    private let bart: BARTG2P?

    static let punctuationTags: Set<NLTag> = Set([
        .openQuote, .closeQuote, .openParenthesis, .closeParenthesis, .punctuation,
        .sentenceTerminator, .otherPunctuation,
    ])
    static let punctuations: Set<Character> = Set(";:,.!?\u{2014}\u{2026}\u{201C}\u{201D}\u{201E}")

    static let punctuationTagPhonemes: [String: String] = [
        "``": String(UnicodeScalar(8220)!),
        "\"\"": String(UnicodeScalar(8221)!),
        "''": String(UnicodeScalar(8221)!),
    ]

    static let nonQuotePunctuations: Set<Character> = Set(
        punctuations.filter { !"\u{201C}\u{201D}\u{201E}".contains($0) })
    static let vowels: Set<Character> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ")
    static let consonants: Set<Character> = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ")
    static let subTokenJunks: Set<Character> = Set("',-._\u{2018}\u{2019}/")
    static let stresses = "ˌˈ"
    static let primaryStress = stresses[stresses.index(stresses.startIndex, offsetBy: 1)]
    static let secondaryStress = stresses[stresses.index(stresses.startIndex, offsetBy: 0)]
    static let subtokenizeRegexPattern =
        // swiftlint:disable:next line_length
        #"^[''']+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|[''']{2,}|\p{L}*?(?:[''']\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:[''']\p{L})*|[^-_\p{L}'''\d]|[''']+$"#
    // swiftlint:disable force_try
    static let subtokenizeRegex = try! NSRegularExpression(
        pattern: EnglishG2P.subtokenizeRegexPattern, options: [])
    /// Regex for splitting CamelCase and compound words into sub-parts.
    private static let camelSplitRegex = try! NSRegularExpression(
        pattern: #"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]{2,}$|[A-Z][a-z]*|[a-z]+"#, options: [])
    private static let linkRegex = try! NSRegularExpression(
        pattern: #"\[([^\]]+)\]\(([^\)]*)\)"#, options: [])
    // swiftlint:enable force_try

    private static let dipthongs: Set<Character> = Set("AIOQWYʤʧ")

    struct PreprocessFeature {
        enum Value {
            case int(Int)
            case double(Double)
            case string(String)
        }

        let value: Value
        let tokenRange: Range<String.Index>
    }

    init(british: Bool = false, unk: String = "❓") {
        self.british = british
        self.tagger = NLTagger(tagSchemes: [.nameTypeOrLexicalClass])
        self.lexicon = Lexicon(british: british)
        self.unk = unk

        self.bart = BARTG2P.fromBundle()
    }

    private func tokenContext(_ ctx: TokenContext, ps: String?, token: MToken) -> TokenContext {
        var vowel = ctx.futureVowel

        if let ps = ps {
            for c in ps {
                if EnglishG2P.nonQuotePunctuations.contains(c) {
                    vowel = nil
                    break
                }

                if EnglishG2P.vowels.contains(c) {
                    vowel = true
                    break
                }

                if EnglishG2P.consonants.contains(c) {
                    vowel = false
                    break
                }
            }
        }
        let futureTo =
            (token.text == "to" || token.text == "To")
            || (token.text == "TO" && (token.tag == .particle || token.tag == .preposition))
        return TokenContext(futureVowel: vowel, futureTo: futureTo)
    }

    func stressWeight(_ phonemes: String?) -> Int {
        guard let phonemes else { return 0 }
        return phonemes.reduce(0) { sum, character in
            sum + (Self.dipthongs.contains(character) ? 2 : 1)
        }
    }

    private func resolveTokens(_ tokens: inout [MToken]) {
        let text =
            tokens.dropLast().map { $0.text + $0.whitespace }.joined()
            + (tokens.last?.text ?? "")
        let prespace =
            text.contains(" ") || text.contains("/")
            || Set(
                text.compactMap { c -> Int? in
                    if EnglishG2P.subTokenJunks.contains(c) { return nil }

                    if c.isLetter { return 0 }
                    if c.isNumber { return 1 }
                    return 2
                }
            ).count > 1

        for i in 0..<tokens.count {
            if tokens[i].phonemes == nil {
                if i == tokens.count - 1, let last = tokens[i].text.last,
                    EnglishG2P.nonQuotePunctuations.contains(last)
                {
                    tokens[i].phonemes = tokens[i].text
                    tokens[i].meta.rating = 3
                } else if tokens[i].text.allSatisfy({
                    EnglishG2P.subTokenJunks.contains($0)
                }) {
                    tokens[i].phonemes = nil
                    tokens[i].meta.rating = 3
                }
            } else if i > 0 {
                tokens[i].meta.prespace = prespace
            }
        }

        guard !prespace else { return }

        var indices: [(Bool, Int, Int)] = []
        for (i, tk) in tokens.enumerated() {
            if let ps = tk.phonemes, !ps.isEmpty {
                indices.append((ps.contains(Lexicon.primaryStress), stressWeight(ps), i))
            }
        }
        if indices.count == 2, tokens[indices[0].2].text.count == 1 {
            let i = indices[1].2
            tokens[i].phonemes = Lexicon.applyStress(tokens[i].phonemes, stress: -0.5)
            return
        } else if indices.count < 2
            || indices.map({ $0.0 ? 1 : 0 }).reduce(0, +) <= (indices.count + 1) / 2
        {
            return
        }
        indices.sort { ($0.0 ? 1 : 0, $0.1) < ($1.0 ? 1 : 0, $1.1) }
        let cut = indices.prefix(indices.count / 2)

        for x in cut {
            let i = x.2
            tokens[i].phonemes = Lexicon.applyStress(tokens[i].phonemes, stress: -0.5)
        }
    }

    typealias PreprocessTuple = (text: String, tokens: [String], features: [PreprocessFeature])

    private func preprocess(text: String) -> PreprocessTuple {
        var result = ""
        var tokens: [String] = []
        var features: [PreprocessFeature] = []

        let input = text.trimmingCharacters(in: .whitespacesAndNewlines)
        var lastEnd = input.startIndex
        let ns = input as NSString
        let fullRange = NSRange(location: 0, length: ns.length)

        Self.linkRegex.enumerateMatches(in: input, options: [], range: fullRange) { match, _, _ in
            guard let m = match else { return }

            let range = m.range
            let start = input.index(input.startIndex, offsetBy: range.location)
            let end = input.index(start, offsetBy: range.length)

            result += String(input[lastEnd..<start])
            tokens.append(
                contentsOf: String(input[lastEnd..<start]).split(separator: " ").map(
                    String.init))

            let grapheme = ns.substring(with: m.range(at: 1))
            let phoneme = ns.substring(with: m.range(at: 2))

            let tokenStartIndex = result.endIndex
            result += grapheme
            let tokenRange = tokenStartIndex..<result.endIndex

            if let intValue = Int(phoneme) {
                features.append(
                    PreprocessFeature(value: .int(intValue), tokenRange: tokenRange))
            } else if ["0.5", "+0.5"].contains(phoneme) {
                features.append(
                    PreprocessFeature(value: .double(0.5), tokenRange: tokenRange))
            } else if phoneme == "-0.5" {
                features.append(
                    PreprocessFeature(value: .double(-0.5), tokenRange: tokenRange))
            } else if phoneme.count > 1 && phoneme.first == "/" && phoneme.last == "/" {
                features.append(
                    PreprocessFeature(
                        value: .string(String(phoneme.dropLast())), tokenRange: tokenRange))
            } else if phoneme.count > 1 && phoneme.first == "#" && phoneme.last == "#" {
                features.append(
                    PreprocessFeature(
                        value: .string(String(phoneme.dropLast())), tokenRange: tokenRange))
            }

            tokens.append(grapheme)
            lastEnd = end
        }

        if lastEnd < input.endIndex {
            result += String(input[lastEnd...])
            tokens.append(
                contentsOf: String(input[lastEnd...]).split(separator: " ").map(String.init))
        }

        return (text: result, tokens: tokens, features: features)
    }

    private func tokenize(preprocessedText: PreprocessTuple) -> [MToken] {
        var mutableTokens: [MToken] = []

        tagger.string = preprocessedText.text
        tagger.setLanguage(
            .english,
            range: preprocessedText.text.startIndex..<preprocessedText.text.endIndex)
        let options: NLTagger.Options = []
        tagger.enumerateTags(
            in: preprocessedText.text.startIndex..<preprocessedText.text.endIndex,
            unit: .word,
            scheme: .nameTypeOrLexicalClass,
            options: options
        ) { tag, tokenRange in
            if let tag = tag {
                let word = String(preprocessedText.text[tokenRange])
                if tag == .whitespace, let lastToken = mutableTokens.last {
                    lastToken.whitespace = word
                } else {
                    mutableTokens.append(
                        MToken(
                            text: word, tokenRange: tokenRange, tag: tag, whitespace: ""))
                }
            }

            return true
        }

        for feature in preprocessedText.features {
            for token in mutableTokens {
                if token.tokenRange.contains(feature.tokenRange)
                    || feature.tokenRange.contains(token.tokenRange)
                {
                    switch feature.value {
                    case .int(let int):
                        token.meta.stress = Double(int)
                    case .double(let double):
                        token.meta.stress = double
                    case .string(let string):
                        if string.hasPrefix("/") {
                            token.meta.is_head = true
                            token.phonemes = String(string.dropFirst())
                            token.meta.rating = 5
                        } else if string.hasPrefix("#") {
                            token.meta.num_flags = String(string.dropFirst())
                        }
                    }
                }
            }
        }

        return mutableTokens
    }

    func mergeTokens(_ tokens: [MToken], unk: String? = nil) -> MToken {
        let stressSet = Set(tokens.compactMap { $0.meta.stress })
        let currencySet = Set(tokens.compactMap { $0.meta.currency })
        let ratings: Set<Int?> = Set(tokens.map { $0.meta.rating })

        var phonemes: String? = nil
        if let unk {
            var phonemeBuilder = ""
            for token in tokens {
                if token.meta.prespace,
                    !phonemeBuilder.isEmpty,
                    !(phonemeBuilder.last?.isWhitespace ?? false),
                    token.phonemes?.isEmpty == false
                {
                    phonemeBuilder += " "
                }
                phonemeBuilder += token.phonemes ?? unk
            }
            phonemes = phonemeBuilder
        }

        let mergedText =
            tokens.dropLast().map { $0.text + $0.whitespace }.joined()
            + (tokens.last?.text ?? "")

        func score(_ t: MToken) -> Int {
            return t.text.reduce(0) {
                $0 + (String($1) == String($1).lowercased() ? 1 : 2)
            }
        }
        let tagSource = tokens.max(by: { score($0) < score($1) })

        let tokenRangeStart = tokens.first!.tokenRange.lowerBound
        let tokenRangeEnd = tokens.last!.tokenRange.upperBound
        let flagChars = Set(tokens.flatMap { Array($0.meta.num_flags) })

        return MToken(
            text: mergedText,
            tokenRange: Range<String.Index>(
                uncheckedBounds: (lower: tokenRangeStart, upper: tokenRangeEnd)),
            tag: tagSource?.tag,
            whitespace: tokens.last?.whitespace ?? "",
            phonemes: phonemes,
            start_ts: tokens.first?.start_ts,
            end_ts: tokens.last?.end_ts,
            underscore: Underscore(
                is_head: tokens.first?.meta.is_head ?? false,
                alias: nil,
                stress: (stressSet.count == 1 ? stressSet.first : nil),
                currency: currencySet.max(),
                num_flags: String(flagChars.sorted()),
                prespace: tokens.first?.meta.prespace ?? false,
                rating: ratings.contains(where: { $0 == nil })
                    ? nil : ratings.compactMap { $0 }.min()
            )
        )
    }

    func foldLeft(_ tokens: [MToken]) -> [MToken] {
        var result: [MToken] = []
        for token in tokens {
            if let last = result.last, !token.meta.is_head {
                _ = result.popLast()
                let merged = mergeTokens([last, token], unk: unk)
                result.append(merged)
            } else {
                result.append(token)
            }
        }
        return result
    }

    func subtokenize(word: String) -> [String] {
        let nsString = word as NSString
        let range = NSRange(location: 0, length: nsString.length)
        let matches = EnglishG2P.subtokenizeRegex.matches(in: word, options: [], range: range)

        return matches.map { match in
            nsString.substring(with: match.range)
        }
    }

    enum RetokenizedItem {
        case single(MToken)
        case compound([MToken])
    }

    // swiftlint:disable:next function_body_length
    func retokenize(_ tokens: [MToken]) -> [RetokenizedItem] {
        var words: [RetokenizedItem] = []
        var currency: String? = nil

        for (i, token) in tokens.enumerated() {
            let needsSplit = (token.meta.alias == nil && token.phonemes == nil)
            var subtokens: [MToken] = []
            if needsSplit {
                let parts = subtokenize(word: token.text)
                subtokens = parts.map { part in
                    let t = MToken(copying: token)
                    t.text = part
                    t.whitespace = ""
                    t.meta.is_head = true
                    t.meta.prespace = false
                    return t
                }
            } else {
                subtokens = [token]
            }
            subtokens.last?.whitespace = token.whitespace

            for j in 0..<subtokens.count {
                let token = subtokens[j]

                if token.meta.alias != nil || token.phonemes != nil {
                    // Already resolved
                } else if token.tag == .otherWord, Lexicon.currencies[token.text] != nil {
                    currency = token.text
                    token.phonemes = ""
                    token.meta.rating = 4
                } else if token.tag == .dash
                    || (token.tag == .punctuation && token.text == "–")
                {
                    token.phonemes = "—"
                    token.meta.rating = 3
                } else if let tag = token.tag, EnglishG2P.punctuationTags.contains(tag),
                    !token.text.lowercased().unicodeScalars.allSatisfy({
                        (97...122).contains(Int($0.value))
                    })
                {
                    if let val = EnglishG2P.punctuationTagPhonemes[token.text] {
                        token.phonemes = val
                    } else {
                        token.phonemes = token.text.filter {
                            EnglishG2P.punctuations.contains($0)
                        }
                    }
                    token.meta.rating = 4
                } else if currency != nil {
                    let looksNumeric = token.text.contains(where: { $0.isNumber })
                    if token.tag != .number && !looksNumeric {
                        currency = nil
                    } else if j + 1 == subtokens.count
                        && (i + 1 == tokens.count || tokens[i + 1].tag != .number)
                    {
                        token.meta.currency = currency
                    }
                } else if j > 0 && j < subtokens.count - 1 && token.text == "2" {
                    let prev = subtokens[j - 1].text
                    let next = subtokens[j + 1].text
                    if (prev.last.map { String($0) } ?? ""
                        + (next.first.map { String($0) } ?? "")).allSatisfy({
                            $0.isLetter
                        })
                        || (prev == "-" && next == "-")
                    {
                        token.meta.alias = "to"
                    }
                }

                // Re-tag otherWord tokens that look numeric (e.g. "98.6", "19.99")
                // so they flow through the number handling path in transcribe()
                if token.tag == .otherWord,
                    token.text.first?.isNumber == true,
                    token.text.allSatisfy({ $0.isNumber || $0 == "." || $0 == "," })
                {
                    token.tag = .number
                }

                if token.meta.alias != nil || token.phonemes != nil {
                    words.append(.single(token))
                } else if case .compound(let last) = words.last,
                    last.last?.whitespace.isEmpty == true
                {
                    var arr = last
                    token.meta.is_head = false
                    arr.append(token)
                    _ = words.popLast()
                    words.append(.compound(arr))
                } else {
                    if token.whitespace.isEmpty { words.append(.compound([token])) } else {
                        words.append(.single(token))
                    }
                }
            }
        }

        return words.map { item in
            if case .compound(let arr) = item, arr.count == 1 {
                return .single(arr[0])
            }
            return item
        }
    }

    // MARK: - CamelCase Fallback (replaces BART neural network)

    /// Split a word on CamelCase / compound boundaries.
    /// "AVFoundation" → ["AV", "Foundation"], "viewDidLoad" → ["view", "Did", "Load"]
    private func splitCamelCase(_ word: String) -> [String] {
        let ns = word as NSString
        let range = NSRange(location: 0, length: ns.length)
        let matches = Self.camelSplitRegex.matches(in: word, options: [], range: range)
        let parts = matches.map { ns.substring(with: $0.range) }
        return parts.isEmpty ? [word] : parts
    }

    /// OOV fallback: CamelCase splitting with per-part resolution.
    ///
    /// Each part is resolved independently: lexicon → BART → letter spelling.
    /// "AVKubernetesPlayer" → "AV"(spell) + "Kubernetes"(BART) + "Player"(lexicon)
    private func fallback(_ word: MToken) -> (phoneme: String?, rating: Int?) {
        let text = word.text

        let parts = splitCamelCase(text)
        if parts.count > 1 {
            let fragments = parts.map { resolvePart($0) }
            let joined = fragments.map(\.0).joined(separator: " ")
            let minRating = fragments.map(\.1).min() ?? 1
            return (joined, minRating)
        }

        return resolvePart(text)
    }

    /// Resolve a single word/part through the full fallback chain:
    /// lexicon → letter spelling (for acronyms) → BART → letter spelling → raw text.
    private func resolvePart(_ text: String) -> (String, Int) {
        if let ph = lexicon.phonemesForWord(text) {
            return (ph, 3)
        }

        // All-uppercase short strings are likely acronyms — spell them out
        if text.count <= 4, text == text.uppercased(), text.allSatisfy({ $0.isLetter }) {
            let nnp = lexicon.getNNP(text)
            if let ph = nnp.0 { return (ph, nnp.1 ?? 2) }
        }

        if let result = bart?.predict(text.lowercased()) {
            return (result, 2)
        }

        let nnp = lexicon.getNNP(text)
        if let phoneme = nnp.phoneme {
            return (phoneme, nnp.rating ?? 2)
        }

        return (unk, 1)
    }

    // MARK: - Main Pipeline

    // swiftlint:disable:next function_body_length
    func phonemize(text: String, performPreprocess: Bool = true) -> (String, [MToken]) {
        let pre: PreprocessTuple
        if performPreprocess {
            pre = self.preprocess(text: text)
        } else {
            pre = (text: text, tokens: [], features: [])
        }

        var tokens = tokenize(preprocessedText: pre)
        tokens = foldLeft(tokens)

        let words = retokenize(tokens)

        var ctx = TokenContext()
        for i in stride(from: words.count - 1, through: 0, by: -1) {
            switch words[i] {
            case .single(let w):
                if w.phonemes == nil {
                    let out = lexicon.transcribe(w, ctx: ctx)
                    w.phonemes = out.0
                    w.meta.rating = out.1
                }

                if w.phonemes == nil {
                    let out = fallback(w)
                    w.phonemes = out.0
                    w.meta.rating = out.1
                }

                ctx = tokenContext(ctx, ps: w.phonemes, token: w)

            case .compound(var arr):
                var left = 0
                var right = arr.count
                var shouldFallback = false
                while left < right {
                    let hasFixed = arr[left..<right].contains {
                        $0.meta.alias != nil || $0.phonemes != nil
                    }
                    let token: MToken? =
                        hasFixed ? nil : mergeTokens(Array(arr[left..<right]))
                    let res: (String?, Int?) =
                        (token == nil) ? (nil, nil) : lexicon.transcribe(token!, ctx: ctx)

                    if let phonemes = res.0 {
                        arr[left].phonemes = phonemes
                        arr[left].meta.rating = res.1
                        for j in (left + 1)..<right {
                            arr[j].phonemes = ""
                            arr[j].meta.rating = res.1
                        }
                        ctx = tokenContext(ctx, ps: phonemes, token: token!)
                        right = left
                        left = 0
                    } else if left + 1 < right {
                        left += 1
                    } else {
                        right -= 1
                        let last = arr[right]
                        if last.phonemes == nil {
                            if last.text.allSatisfy({
                                EnglishG2P.subTokenJunks.contains($0)
                            }) {
                                last.phonemes = ""
                                last.meta.rating = 3
                            } else {
                                shouldFallback = true
                                break
                            }
                        }
                        left = 0
                        arr[right] = last
                    }
                }

                if shouldFallback {
                    let token = mergeTokens(arr)
                    let first = arr[0]
                    let out = fallback(token)
                    first.phonemes = out.0
                    first.meta.rating = out.1
                    arr[0] = first
                    if arr.count > 1 {
                        for j in 1..<arr.count {
                            arr[j].phonemes = ""
                            arr[j].meta.rating = out.1
                        }
                    }
                } else {
                    resolveTokens(&arr)
                }
            }
        }

        let finalTokens: [MToken] = words.map { item in
            switch item {
            case .single(let token):
                return token
            case .compound(let arr):
                return mergeTokens(arr, unk: self.unk)
            }
        }

        for i in 0..<finalTokens.count {
            if var ps = finalTokens[i].phonemes, !ps.isEmpty {
                ps = ps.replacingOccurrences(of: "ɾ", with: "T")
                    .replacingOccurrences(of: "ʔ", with: "t")
                finalTokens[i].phonemes = ps
            }
        }

        let result = finalTokens.map { ($0.phonemes ?? self.unk) + $0.whitespace }.joined()
        return (result, finalTokens)
    }
}
