import XCTest
@testable import SPZ

final class SPZTests: XCTestCase {
    func testLoadSPZ() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "hornedlizard", withExtension: "spz"))
        let cloud = try GaussianCloud.load(from: url)
        
        XCTAssertGreaterThan(cloud.numPoints, 0)
        XCTAssertEqual(cloud.positions.count, cloud.numPoints * 3)
        XCTAssertEqual(cloud.scales.count, cloud.numPoints * 3)
        XCTAssertEqual(cloud.rotations.count, cloud.numPoints * 4)
        XCTAssertEqual(cloud.alphas.count, cloud.numPoints)
        XCTAssertEqual(cloud.colors.count, cloud.numPoints * 3)
        
        let shDim = dimForDegree(cloud.shDegree)
        XCTAssertEqual(cloud.sh.count, cloud.numPoints * shDim * 3)
    }
    
    func testSaveAndLoadSPZ() throws {
        // Load original file
        let originalUrl = try XCTUnwrap(Bundle.module.url(forResource: "hornedlizard", withExtension: "spz"))
        let originalCloud = try GaussianCloud.load(from: originalUrl)
        
        // Save to temporary file
        let tempUrl = FileManager.default.temporaryDirectory.appendingPathComponent("test.spz")
        try GaussianCloud.save(originalCloud, to: tempUrl)
        
        // Load back and compare
        let loadedCloud = try GaussianCloud.load(from: tempUrl)
        
        XCTAssertEqual(loadedCloud.numPoints, originalCloud.numPoints)
        XCTAssertEqual(loadedCloud.shDegree, originalCloud.shDegree)
        XCTAssertEqual(loadedCloud.antialiased, originalCloud.antialiased)
        
        // Compare arrays with small epsilon for floating point differences
        let epsilon: Float = 1e-5
        
        for i in 0..<originalCloud.positions.count {
            XCTAssertEqual(loadedCloud.positions[i], originalCloud.positions[i], accuracy: epsilon)
        }
        
        for i in 0..<originalCloud.scales.count {
            XCTAssertEqual(loadedCloud.scales[i], originalCloud.scales[i], accuracy: epsilon)
        }
        
        for i in 0..<originalCloud.rotations.count {
            XCTAssertEqual(loadedCloud.rotations[i], originalCloud.rotations[i], accuracy: epsilon)
        }
        
        for i in 0..<originalCloud.alphas.count {
            XCTAssertEqual(loadedCloud.alphas[i], originalCloud.alphas[i], accuracy: epsilon)
        }
        
        for i in 0..<originalCloud.colors.count {
            XCTAssertEqual(loadedCloud.colors[i], originalCloud.colors[i], accuracy: epsilon)
        }
        
        for i in 0..<originalCloud.sh.count {
            XCTAssertEqual(loadedCloud.sh[i], originalCloud.sh[i], accuracy: epsilon)
        }
        
        // Clean up
        try? FileManager.default.removeItem(at: tempUrl)
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