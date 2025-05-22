import Foundation

/// A point cloud composed of Gaussians. Each gaussian is represented by:
/// - xyz position
/// - xyz scales (on log scale, compute exp(x) to get scale factor)
/// - xyzw quaternion
/// - alpha (before sigmoid activation, compute sigmoid(a) to get alpha value between 0 and 1)
/// - rgb color (as SH DC component, compute 0.5 + 0.282095 * x to get color value between 0 and 1)
/// - 0 to 45 spherical harmonics coefficients
public struct GaussianCloud {
    /// Total number of points (gaussians) in this splat
    public var numPoints: Int = 0
    
    /// Degree of spherical harmonics for this splat.
    /// Valid values are 0 through 3, where:
    /// - 0: No spherical harmonics (constant color)
    /// - 1: 9 coefficients (3 coeffs x 3 channels)
    /// - 2: 24 coefficients (8 coeffs x 3 channels)
    /// - 3: 45 coefficients (15 coeffs x 3 channels)
    public var shDegree: Int = 0
    
    /// Whether the gaussians should be rendered in antialiased mode (mip splatting)
    public var antialiased: Bool = false
    
    /// XYZ positions for each gaussian. Array size is numPoints * 3.
    /// Values are stored in world space coordinates.
    public var positions: [Float] = []
    
    /// XYZ scales for each gaussian. Array size is numPoints * 3.
    /// Values are stored on log scale - compute exp(x) to get the actual scale factor.
    public var scales: [Float] = []
    
    /// XYZW quaternion rotations for each gaussian. Array size is numPoints * 4.
    /// Values represent the orientation of each gaussian in world space.
    public var rotations: [Float] = []
    
    /// Alpha values for each gaussian. Array size is numPoints.
    /// Values are stored before sigmoid activation - compute sigmoid(a) to get alpha value between 0 and 1.
    public var alphas: [Float] = []
    
    /// RGB colors for each gaussian. Array size is numPoints * 3.
    /// Values are stored as SH DC components - compute 0.5 + 0.282095 * x to get color value between 0 and 1.
    public var colors: [Float] = []
    
    /// Spherical harmonics coefficients. Array size depends on shDegree:
    /// - shDegree 0: Empty array
    /// - shDegree 1: numPoints * 9 (3 coeffs x 3 channels)
    /// - shDegree 2: numPoints * 24 (8 coeffs x 3 channels)
    /// - shDegree 3: numPoints * 45 (15 coeffs x 3 channels)
    /// The color channel is the inner (fastest varying) axis, and the coefficient is the outer
    /// (slower varying) axis.
    public var sh: [Float] = []
    
    /// Creates an empty gaussian cloud
    public init() {}
    
    /// Rotates the GaussianCloud by 180 degrees about the x axis (converts from RUB to RDF coordinates
    /// and vice versa). This is performed in-place.
    public mutating func rotate180DegAboutX() {
        // Rotate positions
        for i in stride(from: 0, to: positions.count, by: 3) {
            positions[i + 1] = -positions[i + 1]
            positions[i + 2] = -positions[i + 2]
        }
        
        // Rotate quaternions
        for i in stride(from: 0, to: rotations.count, by: 4) {
            let x = rotations[i]
            let y = rotations[i + 1]
            let z = rotations[i + 2]
            let w = rotations[i + 3]
            let s: Float = x < 0.0 ? -1.0 : 1.0
            rotations[i] = -s * w
            rotations[i + 1] = s * z
            rotations[i + 2] = -s * y
            rotations[i + 3] = s * x
        }
        
        // Rotate spherical harmonics
        let coeffsToInvert: [Int] = [0, 1, 3, 6, 8, 10, 11, 13]
        let numCoeffs = sh.count / 3
        let numCoeffsPerPoint = numCoeffs / numPoints
        
        for i in stride(from: 0, to: numCoeffs, by: numCoeffsPerPoint) {
            for j in coeffsToInvert {
                guard j < numCoeffsPerPoint else { break }
                let idx = (i + j) * 3
                sh[idx + 0] = -sh[idx + 0]
                sh[idx + 1] = -sh[idx + 1]
                sh[idx + 2] = -sh[idx + 2]
            }
        }
    }
    
    /// Calculates the median volume of all gaussians in the cloud.
    /// Returns 0.01 if the cloud is empty.
    public func medianVolume() -> Float {
        guard numPoints > 0 else { return 0.01 }
        
        // The volume of an ellipsoid is 4/3 * pi * x * y * z, where x, y, and z are the radii on each
        // axis. Scales are stored on a log scale, and exp(x) * exp(y) * exp(z) = exp(x + y + z). So we
        // can sort by value = (x + y + z) and compute volume = 4/3 * pi * exp(value) later.
        var scaleSums: [Float] = []
        for i in stride(from: 0, to: scales.count, by: 3) {
            let sum = scales[i] + scales[i + 1] + scales[i + 2]
            scaleSums.append(sum)
        }
        
        scaleSums.sort()
        let median = scaleSums[scaleSums.count / 2]
        return (Float.pi * 4 / 3) * exp(median)
    }
}

// MARK: - Math Types

/// 3D Vector type using SIMD for efficient calculations
public typealias Vec3f = SIMD3<Float>

/// Quaternion type (w, x, y, z) using SIMD for efficient calculations
public typealias Quat4f = SIMD4<Float>