// swift-tools-version:5.7
import PackageDescription

let package = Package(
    name: "SPZ",
    platforms: [
        .iOS(.v13),
        .macOS(.v11)
    ],
    products: [
        .library(
            name: "SPZ",
            targets: ["SPZ"]),
        .executable(
            name: "spz-tool",
            targets: ["spz-tool"]),
    ],
    dependencies: [
        // No external dependencies required
    ],
    targets: [
        .target(
            name: "SPZ",
            dependencies: [],
            linkerSettings: [
                .linkedLibrary("z")  // Link with zlib
            ]),
        .executableTarget(
            name: "spz-tool",
            dependencies: ["SPZ"]),
        .testTarget(
            name: "SPZTests",
            dependencies: ["SPZ"],
            resources: [
                .process("TestResources")
            ]),
    ]
) 