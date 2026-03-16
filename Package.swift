// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KokoroTTS",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        .executable(name: "kokoro-say", targets: ["CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Jud/swift-bart-g2p.git", from: "0.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "KokoroTTS",
            dependencies: [
                .product(name: "BARTG2P", package: "swift-bart-g2p"),
            ],
            path: "Sources/KokoroTTS",
            resources: [
                .process("Resources"),
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "KokoroTTS",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/CLI"
        ),
        .testTarget(
            name: "KokoroTTSTests",
            dependencies: ["KokoroTTS"],
            path: "Tests/KokoroTTSTests"
        ),
    ]
)
