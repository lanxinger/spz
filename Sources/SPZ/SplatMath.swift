import Foundation
import simd

// MARK: - Vector Operations
extension Vec3f {
    /// Returns a normalized version of the vector (magnitude of 1)
    var normalized: Vec3f {
        self / simd_length(self)
    }
    
    /// Returns the squared magnitude of the vector.
    /// This is more efficient than computing the actual magnitude when only comparing distances.
    var squaredNorm: Float {
        simd_dot(self, self)
    }
}

// MARK: - Quaternion Operations
extension Quat4f {
    /// Returns a normalized version of the quaternion (magnitude of 1)
    var normalized: Quat4f {
        self / simd_length(self)
    }
    
    /// Rotates a vector by this quaternion.
    /// - Parameter v: The vector to rotate
    /// - Returns: The rotated vector
    func rotate(_ v: Vec3f) -> Vec3f {
        // Use SIMD operations for faster quaternion rotation
        let q = self
        let u = SIMD3<Float>(q.x, q.y, q.z)
        let uv = simd_cross(u, v)
        let uuv = simd_cross(u, uv)
        return v + ((uv * q.w) + uuv) * 2
    }
}

// MARK: - Math Utilities
/// Creates a quaternion from an axis-angle representation.
/// - Parameter scaledAxis: A vector whose direction represents the axis of rotation and whose
///   magnitude represents the angle of rotation in radians.
/// - Returns: A normalized quaternion representing the rotation.
func axisAngleQuat(_ scaledAxis: Vec3f) -> Quat4f {
    let thetaSquared = scaledAxis.squaredNorm
    
    // For points not at the origin, the full conversion is numerically stable
    if thetaSquared > 0.0 {
        let theta = sqrt(thetaSquared)
        let halfTheta = theta * 0.5
        let k = sin(halfTheta) / theta
        return Quat4f(cos(halfTheta), scaledAxis.x * k, scaledAxis.y * k, scaledAxis.z * k).normalized
    }
    
    // If thetaSquared is 0, then we will get NaNs when dividing by theta.
    // By approximating with a Taylor series, and truncating at one term, 
    // the value will be computed correctly.
    let k: Float = 0.5
    return Quat4f(1.0, scaledAxis.x * k, scaledAxis.y * k, scaledAxis.z * k).normalized
} 