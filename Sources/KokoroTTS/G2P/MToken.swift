// Originally from MLXUtilsLibrary by mlalma, Apache License 2.0

import Foundation
import NaturalLanguage

class Underscore {
    var is_head: Bool
    var alias: String?
    var stress: Double?
    var currency: String?
    var num_flags: String
    var prespace: Bool
    var rating: Int?

    init(
        is_head: Bool = true,
        alias: String? = nil,
        stress: Double? = nil,
        currency: String? = nil,
        num_flags: String = "",
        prespace: Bool = false,
        rating: Int? = nil
    ) {
        self.is_head = is_head
        self.alias = alias
        self.stress = stress
        self.currency = currency
        self.num_flags = num_flags
        self.prespace = prespace
        self.rating = rating
    }

    convenience init(copying other: Underscore) {
        self.init(
            is_head: other.is_head,
            alias: other.alias,
            stress: other.stress,
            currency: other.currency,
            num_flags: other.num_flags,
            prespace: other.prespace,
            rating: other.rating)
    }
}

extension Underscore: CustomStringConvertible {
    var description: String {
        let mirror = Mirror(reflecting: self)
        let props = mirror.children.compactMap { child -> String? in
            guard let label = child.label else { return nil }
            return "\(label): \(String(describing: child.value))"
        }.joined(separator: ", ")
        return "\(type(of: self))(\(props))"
    }
}

class MToken {
    var text: String
    var tokenRange: Range<String.Index>
    var tag: NLTag?
    var whitespace: String
    var phonemes: String?
    var start_ts: Double?
    var end_ts: Double?
    var meta: Underscore

    init(
        text: String,
        tokenRange: Range<String.Index>,
        tag: NLTag? = nil,
        whitespace: String,
        phonemes: String? = nil,
        start_ts: Double? = nil,
        end_ts: Double? = nil,
        underscore: Underscore = Underscore()
    ) {
        self.text = text
        self.tokenRange = tokenRange
        self.tag = tag
        self.whitespace = whitespace
        self.phonemes = phonemes
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.meta = underscore
    }

    convenience init(copying other: MToken) {
        self.init(
            text: other.text,
            tokenRange: other.tokenRange,
            tag: other.tag,
            whitespace: other.whitespace,
            phonemes: other.phonemes,
            start_ts: other.start_ts,
            end_ts: other.end_ts,
            underscore: Underscore(copying: other.meta))
    }
}

extension MToken: CustomStringConvertible {
    var description: String {
        let mirror = Mirror(reflecting: self)
        let props = mirror.children.compactMap { child -> String? in
            guard let label = child.label else { return nil }
            return "\(label): \(String(describing: child.value))"
        }.joined(separator: ", ")
        return "\(type(of: self))(\(props))"
    }
}
