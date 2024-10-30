import Foundation
import SPZ

func printUsage() {
    print("Usage:")
    print("  spz-tool info <file.spz>     - Print information about SPZ file")
    print("  spz-tool convert <in> <out>  - Convert between .spz and .ply formats")
}

guard CommandLine.arguments.count >= 3 else {
    printUsage()
    exit(1)
}

let command = CommandLine.arguments[1]
let inputPath = CommandLine.arguments[2]
let debug = CommandLine.arguments.contains("--debug")

do {
    switch command {
    case "info":
        let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        if debug {
            print("[SPZ] File size: \(data.count) bytes")
        }
        
        guard let decompressed = decompressGzipped(data) else {
            throw SPZError.decompressionError
        }
        
        if debug {
            print("[SPZ] Decompressed size: \(decompressed.count) bytes")
            print("[SPZ] Attempting to deserialize...")
        }
        
        let packed = try PackedGaussians.deserialize(decompressed)
        
        if debug {
            print("[SPZ] Successfully deserialized packed data:")
            print("  - Number of points: \(packed.numPoints)")
            print("  - SH degree: \(packed.shDegree)")
            print("  - Uses float16: \(packed.usesFloat16)")
            print("  - Array sizes:")
            print("    - positions: \(packed.positions.count)")
            print("    - scales: \(packed.scales.count)")
            print("    - rotations: \(packed.rotations.count)")
            print("    - alphas: \(packed.alphas.count)")
            print("    - colors: \(packed.colors.count)")
            print("    - sh: \(packed.sh.count)")
            print("[SPZ] Attempting to unpack first gaussian...")
        }
        
        let cloud = unpackGaussians(packed)
        
        print("Number of points: \(cloud.numPoints)")
        print("SH degree: \(cloud.shDegree)")
        print("Antialiased: \(cloud.antialiased)")
        print("Median volume: \(cloud.medianVolume())")
        
    case "convert":
        guard CommandLine.arguments.count >= 4 else {
            printUsage()
            exit(1)
        }
        let outputPath = CommandLine.arguments[3]
        let inputUrl = URL(fileURLWithPath: inputPath)
        let outputUrl = URL(fileURLWithPath: outputPath)
        
        print("\r[SPZ] Converting: 0% (Loading)", terminator: "")
        fflush(stdout)
        
        let cloud: GaussianCloud
        if inputPath.hasSuffix(".ply") {
            cloud = try GaussianCloud.loadFromPly(url: inputUrl)
        } else {
            cloud = try GaussianCloud.load(from: inputUrl)
        }
        
        print("\r[SPZ] Converting: 50% (Saving)", terminator: "")
        fflush(stdout)
        
        if outputPath.hasSuffix(".ply") {
            try cloud.saveToPly(url: outputUrl)
        } else {
            try GaussianCloud.save(cloud, to: outputUrl)
        }
        
        print("\r[SPZ] Converting: 100% (Complete)")
        print("Converted \(cloud.numPoints) points")
        
    default:
        printUsage()
        exit(1)
    }
} catch {
    print("\nError: \(error)")
    exit(1)
} 