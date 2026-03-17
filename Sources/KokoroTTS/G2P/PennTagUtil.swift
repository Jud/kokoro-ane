// Originally from MisakiSwift by mlalma, Apache License 2.0

import NaturalLanguage

private let whDeterminers: Set<String> = ["which", "whatever", "whichever"]
private let whPronouns: Set<String> = [
    "who", "whom", "whose", "whoever", "whomever", "what", "whatever", "which", "whichever",
]
private let whAdverbs: Set<String> = ["when", "where", "why", "how"]
private let possessivePronouns: Set<String> = ["my", "your", "his", "her", "its", "our", "their"]
private let auxBe: Set<String> = ["am", "is", "are", "was", "were", "be", "been", "being"]
private let auxDo: Set<String> = ["do", "does", "did"]
private let auxHave: Set<String> = ["have", "has", "had"]
private let subordinatingConjunctions: Set<String> = [
    "because", "although", "though", "if", "while", "when", "whenever", "before", "after",
    "since", "unless", "until", "that", "whether", "as",
]
private let personalPronouns: Set<String> = [
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
]

func pennTag(for nlTag: NLTag, token: String? = nil) -> String {
    let t = token?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let lower = t.lowercased()

    if let punct = pennPunctuationTag(for: nlTag, token: t) { return punct }

    switch nlTag {
    case .noun: return pennNounTag(t, lower: lower)
    case .verb: return pennVerbTag(lower)
    case .adjective: return pennAdjectiveTag(lower)
    case .adverb: return pennAdverbTag(lower)
    case .pronoun: return pennPronounTag(lower)
    case .determiner: return whDeterminers.contains(lower) ? "WDT" : "DT"
    case .preposition: return lower == "to" ? "TO" : "IN"
    case .conjunction: return subordinatingConjunctions.contains(lower) ? "IN" : "CC"
    case .number: return "CD"
    case .interjection: return "UH"
    case .particle: return lower == "to" ? "TO" : "RP"
    case .word, .otherWord: return "FW"
    case .whitespace, .paragraphBreak, .wordJoiner: return "XX"
    case .personalName, .organizationName, .placeName: return "NNP"
    case .classifier, .idiom, .dash: return "FW"
    default: return "XX"
    }
}

func isPersonalPronoun(tag: NLTag, token: String) -> Bool {
    tag == .pronoun && personalPronouns.contains(token.lowercased())
}

// MARK: - Helpers

private func pennPunctuationTag(for nlTag: NLTag, token: String) -> String? {
    if nlTag == .punctuation || nlTag == .sentenceTerminator || nlTag == .otherPunctuation {
        switch token {
        case ",": return ","
        case ".", "!", "?": return "."
        case ":", ";": return ":"
        case "``", "\u{201C}", "\u{201E}", "\"": return "``"
        case "''", "\u{201D}": return "''"
        case "(", "[", "{": return "("
        case ")", "]", "}": return ")"
        case "$": return "$"
        case "#": return "#"
        case "-", "–", "—": return ":"
        default: return nil
        }
    }
    switch nlTag {
    case .openQuote: return "``"
    case .closeQuote: return "''"
    case .openParenthesis: return "("
    case .closeParenthesis: return ")"
    case .punctuation, .sentenceTerminator, .otherPunctuation: return "."
    default: return nil
    }
}

private func pennNounTag(_ token: String, lower: String) -> String {
    guard !token.isEmpty else { return "NN" }
    let capitalized = token.first?.isUppercase == true
    let plural =
        lower.count > 2 && lower.hasSuffix("s")
        && !lower.hasSuffix("ss") && !lower.hasSuffix("'s") && !lower.hasSuffix("\u{2019}s")
    if capitalized && !plural { return "NNP" }
    if capitalized && plural { return "NNPS" }
    if plural { return "NNS" }
    return "NN"
}

private func pennVerbTag(_ lower: String) -> String {
    if auxBe.contains(lower) {
        return lower == "being" ? "VBG" : (lower == "been" ? "VBN" : "VB")
    }
    if auxDo.contains(lower) {
        return lower == "does" ? "VBZ" : (lower == "did" ? "VBD" : "VB")
    }
    if auxHave.contains(lower) {
        return lower == "has" ? "VBZ" : (lower == "had" ? "VBD" : "VB")
    }
    if lower.hasSuffix("ing") { return "VBG" }
    if lower.hasSuffix("ed") { return "VBD" }
    if lower.hasSuffix("en") { return "VBN" }
    if lower.hasSuffix("s") { return "VBZ" }
    return "VB"
}

private func pennAdjectiveTag(_ lower: String) -> String {
    if lower.hasSuffix("er") { return "JJR" }
    if lower.hasSuffix("est") { return "JJS" }
    return "JJ"
}

private func pennAdverbTag(_ lower: String) -> String {
    if whAdverbs.contains(lower) { return "WRB" }
    if lower.hasSuffix("er") { return "RBR" }
    if lower.hasSuffix("est") { return "RBS" }
    return "RB"
}

private func pennPronounTag(_ lower: String) -> String {
    if lower == "'s" || lower == "\u{2019}s" { return "POS" }
    if whPronouns.contains(lower) {
        return lower == "whose" ? "WP$" : "WP"
    }
    if possessivePronouns.contains(lower) { return "PRP$" }
    return "PRP"
}
