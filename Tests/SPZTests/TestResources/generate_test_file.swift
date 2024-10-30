import Foundation
import SPZ

// Create a simple test gaussian cloud
var cloud = GaussianCloud()
cloud.numPoints = 1
cloud.shDegree = 1
cloud.antialiased = false

// Add a single gaussian
cloud.positions = [0.0, 0.0, 0.0]  // Center position
cloud.scales = [0.1, 0.1, 0.1]     // Unit scale
cloud.rotations = [0.0, 0.0, 0.0, 1.0]  // Identity rotation
cloud.alphas = [1.0]  // Fully opaque
cloud.colors = [0.5, 0.5, 0.5]  // Gray color
cloud.sh = Array(repeating: 0.0, count: 9)  // Zero spherical harmonics

// Create directory if it doesn't exist
let fileManager = FileManager.default
let testResourcesPath = "Tests/SPZTests/TestResources"
try? fileManager.createDirectory(atPath: testResourcesPath, withIntermediateDirectories: true)

// Save to test file
let testFilePath = "\(testResourcesPath)/test.spz"
try GaussianCloud.save(cloud, to: URL(fileURLWithPath: testFilePath))
print("Created test file at: \(testFilePath)") 