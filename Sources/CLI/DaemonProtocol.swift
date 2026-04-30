import CBOR
import Foundation
import KokoroCoreML

enum DaemonConfig {
    private static let basePath = "\(NSTemporaryDirectory())kokoro-coreml-\(getuid())"
    static let socketPath = basePath + ".sock"
    static let pidPath = basePath + ".pid"

    /// Increment when the wire format changes.
    static let protocolVersion: Int = 2
}

struct SynthesisRequest: Codable {
    var version: Int = DaemonConfig.protocolVersion
    var text: String
    var voice: String
    var speed: Float
}

struct SynthesisResponse: Codable {
    var version: Int = DaemonConfig.protocolVersion
    var ok: Bool
    var error: String?
    var sampleCount: Int?
    var synthesisTime: Double?
    var phonemes: String?
    var tokenCount: Int?
    var timestamps: [SynthesisTimestamp]?
}

// MARK: - CBOR I/O

enum DaemonIO {
    static func writeMessage<T: Encodable>(_ value: T, to fd: Int32) -> Bool {
        let bytes: [UInt8]
        do {
            bytes = try CBOREncoder().encode(value)
        } catch {
            fputs("CBOR encode error: \(error)\n", stderr)
            return false
        }
        return LengthPrefixedIO.writeBytes(bytes, to: fd)
    }

    static func readMessage<T: Decodable>(_ type: T.Type, from fd: Int32) -> T? {
        guard let bytes = LengthPrefixedIO.readBytes(from: fd) else { return nil }
        do {
            return try CBORDecoder().decode(type, from: bytes)
        } catch {
            fputs("CBOR decode error (\(type)): \(error)\n", stderr)
            return nil
        }
    }
}

// MARK: - Unix Socket Helpers

enum UnixSocket {
    /// Connect to a Unix domain socket. Returns fd on success, -1 on failure.
    static func connect(to path: String) -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return -1 }

        var addr = sockaddr_un()
        guard fillAddress(&addr, path: path) else {
            close(fd)
            return -1
        }

        let result = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.connect(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else {
            close(fd)
            return -1
        }
        return fd
    }

    /// Create and bind a listening Unix domain socket. Returns fd on success, -1 on failure.
    static func bind(to path: String, backlog: Int32 = 8) -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return -1 }

        unlink(path)

        var addr = sockaddr_un()
        guard fillAddress(&addr, path: path) else {
            close(fd)
            return -1
        }

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            close(fd)
            return -1
        }

        guard listen(fd, backlog) == 0 else {
            close(fd)
            unlink(path)
            return -1
        }

        return fd
    }

    private static func fillAddress(_ addr: inout sockaddr_un, path: String) -> Bool {
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else { return false }
        withUnsafeMutableBytes(of: &addr.sun_path) { buf in
            pathBytes.withUnsafeBytes { src in
                buf.copyMemory(from: src)
            }
        }
        return true
    }
}

// MARK: - Length-prefixed I/O

enum LengthPrefixedIO {
    static func writeBytes(_ bytes: [UInt8], to fd: Int32) -> Bool {
        guard writeLengthHeader(bytes.count, to: fd) else { return false }
        return bytes.withUnsafeBytes { writeFully(fd: fd, from: $0.baseAddress!, count: bytes.count) }
    }

    static func writeRawSamples(_ samples: [Float], to fd: Int32) -> Bool {
        let byteCount = samples.count * MemoryLayout<Float>.size
        guard writeLengthHeader(byteCount, to: fd) else { return false }
        return samples.withUnsafeBytes { writeFully(fd: fd, from: $0.baseAddress!, count: byteCount) }
    }

    static func readRawSamples(count: Int, from fd: Int32) -> [Float]? {
        var lengthBE: UInt32 = 0
        let headerRead = withUnsafeMutableBytes(of: &lengthBE) { buf in
            readFully(fd: fd, into: buf.baseAddress!, count: 4)
        }
        guard headerRead else { return nil }
        let byteCount = Int(UInt32(bigEndian: lengthBE))
        guard byteCount == count * MemoryLayout<Float>.size else { return nil }
        var samples = [Float](repeating: 0, count: count)
        let ok = samples.withUnsafeMutableBytes { buf in
            readFully(fd: fd, into: buf.baseAddress!, count: byteCount)
        }
        return ok ? samples : nil
    }

    static func readBytes(from fd: Int32) -> [UInt8]? {
        var lengthBE: UInt32 = 0
        let headerRead = withUnsafeMutableBytes(of: &lengthBE) { buf in
            readFully(fd: fd, into: buf.baseAddress!, count: 4)
        }
        guard headerRead else { return nil }
        let length = Int(UInt32(bigEndian: lengthBE))
        guard length > 0, length < 100_000_000 else { return nil }
        var bytes = [UInt8](repeating: 0, count: length)
        let ok = bytes.withUnsafeMutableBytes { buf in
            readFully(fd: fd, into: buf.baseAddress!, count: length)
        }
        return ok ? bytes : nil
    }

    private static func writeLengthHeader(_ length: Int, to fd: Int32) -> Bool {
        var header = UInt32(length).bigEndian
        return withUnsafeBytes(of: &header) { buf in
            Darwin.write(fd, buf.baseAddress!, 4) == 4
        }
    }

    private static func writeFully(fd: Int32, from ptr: UnsafeRawPointer, count: Int) -> Bool {
        var offset = 0
        while offset < count {
            let n = Darwin.write(fd, ptr + offset, count - offset)
            if n <= 0 { return false }
            offset += n
        }
        return true
    }

    private static func readFully(fd: Int32, into ptr: UnsafeMutableRawPointer, count: Int) -> Bool {
        var offset = 0
        while offset < count {
            let n = Darwin.read(fd, ptr + offset, count - offset)
            if n <= 0 { return false }
            offset += n
        }
        return true
    }
}
