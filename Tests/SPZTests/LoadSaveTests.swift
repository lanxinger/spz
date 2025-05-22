import XCTest
@testable import SPZ

final class LoadSaveTests: XCTestCase {
    
    func testBasicLoadSave() throws {
        // Get path to test file
        let testBundle = Bundle.module
        guard let inputUrl = testBundle.url(forResource: "hornedlizard", withExtension: "spz") else {
            throw XCTSkip("Test file not found - run generate_test_file.swift first")
        }
        
        // Load the file
        print("Loading from: \(inputUrl.path)")
        let cloud = try GaussianCloud.load(from: inputUrl)
        
        // Verify basic properties
        XCTAssertGreaterThan(cloud.numPoints, 0)
        XCTAssertGreaterThanOrEqual(cloud.shDegree, 0)
        XCTAssertLessThanOrEqual(cloud.shDegree, 3)
        
        // Verify array sizes
        XCTAssertEqual(cloud.positions.count, cloud.numPoints * 3)
        XCTAssertEqual(cloud.scales.count, cloud.numPoints * 3)
        XCTAssertEqual(cloud.rotations.count, cloud.numPoints * 4)
        XCTAssertEqual(cloud.alphas.count, cloud.numPoints)
        XCTAssertEqual(cloud.colors.count, cloud.numPoints * 3)
        
        let shDim = dimForDegree(cloud.shDegree)
        XCTAssertEqual(cloud.sh.count, cloud.numPoints * shDim * 3)
        
        // Save to temporary file
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent("test_output.spz")
        try GaussianCloud.save(cloud, to: tempUrl)
        
        // Load back and compare
        let reloadedCloud = try GaussianCloud.load(from: tempUrl)
        
        // Compare properties
        XCTAssertEqual(reloadedCloud.numPoints, cloud.numPoints)
        XCTAssertEqual(reloadedCloud.shDegree, cloud.shDegree)
        XCTAssertEqual(reloadedCloud.antialiased, cloud.antialiased)
        
        // Compare arrays with small epsilon for floating point differences
        let epsilon: Float = 1e-5
        
        // Helper function to compare arrays
        func compareArrays(_ a: [Float], _ b: [Float], name: String) {
            XCTAssertEqual(a.count, b.count, "\(name) arrays have different lengths")
            for i in 0..<a.count {
                XCTAssertEqual(a[i], b[i], accuracy: epsilon, "\(name) differs at index \(i)")
            }
        }
        
        compareArrays(reloadedCloud.positions, cloud.positions, name: "positions")
        compareArrays(reloadedCloud.scales, cloud.scales, name: "scales")
        compareArrays(reloadedCloud.rotations, cloud.rotations, name: "rotations")
        compareArrays(reloadedCloud.alphas, cloud.alphas, name: "alphas")
        compareArrays(reloadedCloud.colors, cloud.colors, name: "colors")
        compareArrays(reloadedCloud.sh, cloud.sh, name: "spherical harmonics")
        
        // Print some statistics
        print("Successfully loaded and saved \(cloud.numPoints) points")
        print("SH degree: \(cloud.shDegree)")
        print("Median volume: \(cloud.medianVolume())")
        
        // Clean up
        try? FileManager.default.removeItem(at: tempUrl)
    }
    
    func testRotation() throws {
        // Load test file
        let testBundle = Bundle.module
        guard let inputUrl = testBundle.url(forResource: "hornedlizard", withExtension: "spz") else {
            XCTFail("Could not find test file")
            return
        }
        
        var cloud = try GaussianCloud.load(from: inputUrl)
        
        // Save original values
        let originalPositions = cloud.positions
        let originalRotations = cloud.rotations
        let originalSH = cloud.sh
        
        // Rotate 180° about X axis twice (should return to original position)
        cloud.rotate180DegAboutX()
        cloud.rotate180DegAboutX()
        
        // Compare with original values
        let epsilon: Float = 1e-5
        
        for i in 0..<cloud.positions.count {
            XCTAssertEqual(cloud.positions[i], originalPositions[i], accuracy: epsilon,
                          "Position \(i) changed after double rotation")
        }
        
        for i in 0..<cloud.rotations.count {
            XCTAssertEqual(cloud.rotations[i], originalRotations[i], accuracy: epsilon,
                          "Rotation \(i) changed after double rotation")
        }
        
        for i in 0..<cloud.sh.count {
            XCTAssertEqual(cloud.sh[i], originalSH[i], accuracy: epsilon,
                          "SH coefficient \(i) changed after double rotation")
        }
    }
}

private func dimForDegree(_ degree: Int) -> Int {
    switch degree {
    case 0: return 0
    case 1: return 3
    case 2: return 8
    case 3: return 15
    default: return 0
    }
} 