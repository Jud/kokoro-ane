import Foundation

enum DaemonResult {
    case success(SynthesisResponse, [Float])
    case daemonError(String)
    case unavailable
}

enum DaemonClient {
    /// Try to synthesize via the daemon.
    /// Returns .unavailable if daemon isn't running, .daemonError if daemon
    /// returned an error, .success with response + samples on success.
    static func synthesize(_ request: SynthesisRequest) -> DaemonResult {
        let fd = UnixSocket.connect(to: DaemonConfig.socketPath)
        guard fd >= 0 else { return .unavailable }
        defer { close(fd) }

        guard DaemonIO.writeMessage(request, to: fd) else {
            return .daemonError("Failed to send request")
        }

        guard let response = DaemonIO.readMessage(SynthesisResponse.self, from: fd) else {
            return .daemonError("Failed to read response")
        }

        guard response.ok else {
            return .daemonError(response.error ?? "Unknown daemon error")
        }

        guard let sampleCount = response.sampleCount, sampleCount > 0 else {
            return .success(response, [])
        }

        guard let samples = LengthPrefixedIO.readRawSamples(count: sampleCount, from: fd) else {
            return .daemonError("Failed to read audio data")
        }

        return .success(response, samples)
    }

    /// Check if daemon is running by attempting a socket connect.
    static func isRunning() -> Bool {
        let fd = UnixSocket.connect(to: DaemonConfig.socketPath)
        guard fd >= 0 else { return false }
        close(fd)
        return true
    }
}
