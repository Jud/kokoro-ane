import ArgumentParser
import Foundation
import KokoroTTS

nonisolated(unsafe) private var daemonShouldQuit = false

struct Daemon: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Manage the Kokoro TTS daemon for fast inference",
        subcommands: [Start.self, Stop.self, Restart.self, Status.self]
    )

    // MARK: - Start

    struct Start: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Start the daemon (loads models, detaches to background)"
        )

        @Option(name: .long, help: "Model directory path")
        var modelDir: String?

        @Flag(name: .long, help: .hidden)
        var foreground = false

        func run() throws {
            let pidPath = DaemonConfig.pidPath
            let sockPath = DaemonConfig.socketPath

            if foreground {
                try runForeground(pidPath: pidPath, sockPath: sockPath)
                return
            }

            // Check if already running
            if let existingPid = readPID(from: pidPath), isProcessAlive(existingPid) {
                print("Daemon already running (PID \(existingPid))")
                return
            }

            // Clean stale files
            cleanStaleFiles(pidPath: pidPath, sockPath: sockPath)

            // Spawn background child with --foreground flag
            let execPath = CommandLine.arguments[0]
            var args = ["kokoro", "daemon", "start", "--foreground"]
            if let dir = modelDir {
                args += ["--model-dir", dir]
            }

            var cArgs = args.map { strdup($0) } + [nil]
            defer { cArgs.forEach { $0.map { free($0) } } }

            // Set up file actions: redirect stdin/stdout/stderr to /dev/null
            var fileActions: posix_spawn_file_actions_t?
            posix_spawn_file_actions_init(&fileActions)
            defer { posix_spawn_file_actions_destroy(&fileActions) }
            posix_spawn_file_actions_addopen(&fileActions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
            posix_spawn_file_actions_addopen(&fileActions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0)
            posix_spawn_file_actions_addopen(&fileActions, STDERR_FILENO, "/dev/null", O_WRONLY, 0)

            // Set POSIX_SPAWN_SETSID to create new session (detach from terminal)
            var spawnAttr: posix_spawnattr_t?
            posix_spawnattr_init(&spawnAttr)
            defer { posix_spawnattr_destroy(&spawnAttr) }
            posix_spawnattr_setflags(&spawnAttr, Int16(POSIX_SPAWN_SETSID))

            var childPid: pid_t = 0
            let spawnResult = posix_spawn(&childPid, execPath, &fileActions, &spawnAttr, &cArgs, environ)
            guard spawnResult == 0 else {
                fputs("Failed to spawn daemon: \(String(cString: strerror(spawnResult)))\n", stderr)
                throw ExitCode.failure
            }

            // Write PID file immediately
            try String(childPid).write(toFile: pidPath, atomically: true, encoding: .utf8)

            // Wait for daemon to become ready (socket appears)
            var ready = false
            for _ in 0..<100 {  // up to 10 seconds
                Thread.sleep(forTimeInterval: 0.1)
                if DaemonClient.isRunning() {
                    ready = true
                    break
                }
                // Check if child died
                var status: Int32 = 0
                let w = waitpid(childPid, &status, WNOHANG)
                if w > 0 {
                    cleanStaleFiles(pidPath: pidPath, sockPath: sockPath)
                    fputs("Daemon failed to start (exited with status \(status))\n", stderr)
                    throw ExitCode.failure
                }
            }

            if ready {
                print("Kokoro daemon ready (PID \(childPid))")
            } else {
                fputs("Daemon started (PID \(childPid)) but not yet responding\n", stderr)
            }
        }

        private func runForeground(pidPath: String, sockPath: String) throws {
            let dir =
                modelDir.map { URL(fileURLWithPath: $0) } ?? KokoroEngine.defaultModelDirectory
            try CLIModelDownloader.ensureModels(at: dir)
            let engine = try KokoroEngine(modelDirectory: dir)

            // Wait for auto-warmup to complete before accepting connections
            while !engine.isReady {
                Thread.sleep(forTimeInterval: 0.05)
            }

            let serverFd = UnixSocket.bind(to: sockPath)
            guard serverFd >= 0 else { throw ExitCode.failure }

            try String(getpid()).write(toFile: pidPath, atomically: true, encoding: .utf8)

            // Signal handling
            daemonShouldQuit = false
            signal(SIGTERM) { _ in daemonShouldQuit = true }
            signal(SIGINT) { _ in daemonShouldQuit = true }
            signal(SIGPIPE, SIG_IGN)

            // Accept loop
            var pollFd = pollfd(fd: serverFd, events: Int16(POLLIN), revents: 0)

            while !daemonShouldQuit {
                pollFd.revents = 0
                let pollResult = poll(&pollFd, 1, 1000)
                if pollResult <= 0 { continue }

                var clientAddr = sockaddr_un()
                var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
                let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                        accept(serverFd, sockPtr, &clientLen)
                    }
                }
                guard clientFd >= 0 else { continue }

                let thread = Thread {
                    handleClient(fd: clientFd, engine: engine)
                }
                thread.stackSize = 8 * 1024 * 1024
                thread.start()
            }

            // Cleanup
            close(serverFd)
            unlink(sockPath)
            unlink(pidPath)
            // Use _exit to avoid running Swift atexit/deinit handlers that may
            // hang in a detached daemon process (no runloop, no main thread).
            _exit(0)
        }
    }

    // MARK: - Stop

    struct Stop: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Stop the running daemon"
        )

        func run() throws {
            let pidPath = DaemonConfig.pidPath
            let sockPath = DaemonConfig.socketPath

            guard let pid = readPID(from: pidPath) else {
                print("Daemon is not running")
                return
            }

            guard isProcessAlive(pid) else {
                print("Daemon is not running (stale PID file)")
                cleanStaleFiles(pidPath: pidPath, sockPath: sockPath)
                return
            }

            kill(pid, SIGTERM)

            // Wait up to 3 seconds
            for _ in 0..<30 {
                if !isProcessAlive(pid) { break }
                Thread.sleep(forTimeInterval: 0.1)
            }

            if isProcessAlive(pid) {
                kill(pid, SIGKILL)
                Thread.sleep(forTimeInterval: 0.1)
            }

            cleanStaleFiles(pidPath: pidPath, sockPath: sockPath)
            print("Daemon stopped")
        }
    }

    // MARK: - Restart

    struct Restart: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Restart the daemon"
        )

        @Option(name: .long, help: "Model directory path")
        var modelDir: String?

        func run() throws {
            if let pid = readPID(from: DaemonConfig.pidPath), isProcessAlive(pid) {
                let stop = Stop()
                try stop.run()
            }
            var start = Start()
            start.modelDir = modelDir
            try start.run()
        }
    }

    // MARK: - Status

    struct Status: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Check daemon status"
        )

        func run() {
            let pidPath = DaemonConfig.pidPath

            guard let pid = readPID(from: pidPath) else {
                print("Daemon is not running")
                return
            }

            guard isProcessAlive(pid) else {
                print("Daemon is not running (stale PID file)")
                return
            }

            print("Daemon is running (PID \(pid))")
        }
    }
}

// MARK: - Helpers

private func readPID(from path: String) -> pid_t? {
    guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
    return pid_t(contents.trimmingCharacters(in: .whitespacesAndNewlines))
}

private func isProcessAlive(_ pid: pid_t) -> Bool {
    kill(pid, 0) == 0
}

private func cleanStaleFiles(pidPath: String, sockPath: String) {
    unlink(pidPath)
    unlink(sockPath)
}

private func handleClient(fd: Int32, engine: KokoroEngine) {
    defer { close(fd) }

    guard let requestData = LengthPrefixedIO.readMessage(from: fd) else { return }
    guard let request = try? JSONDecoder().decode(SynthesisRequest.self, from: requestData) else {
        let err = SynthesisResponse(ok: false, error: "Invalid request")
        if let data = try? JSONEncoder().encode(err) {
            _ = LengthPrefixedIO.writeMessage(data, to: fd)
        }
        return
    }

    do {
        let result = try engine.synthesize(
            text: request.text, voice: request.voice,
            speed: request.speed, rawAudio: request.raw)

        let response = SynthesisResponse(
            ok: true,
            sampleCount: result.samples.count,
            synthesisTime: result.synthesisTime,
            phonemes: result.phonemes,
            tokenCount: result.tokenCount)

        guard let responseData = try? JSONEncoder().encode(response) else { return }
        guard LengthPrefixedIO.writeMessage(responseData, to: fd) else { return }
        _ = LengthPrefixedIO.writeRawSamples(result.samples, to: fd)
    } catch {
        let errResponse = SynthesisResponse(ok: false, error: error.localizedDescription)
        if let data = try? JSONEncoder().encode(errResponse) {
            _ = LengthPrefixedIO.writeMessage(data, to: fd)
        }
    }
}
