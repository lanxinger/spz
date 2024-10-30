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
        let w = self.w
        let x = self.x
        let y = self.y
        let z = self.z
        
        let x2 = x + x
        let y2 = y + y
        let z2 = z + z
        let wx2 = w * x2
        let wy2 = w * y2
        let wz2 = w * z2
        let xx2 = x * x2
        let xy2 = x * y2
        let xz2 = x * z2
        let yy2 = y * y2
        let yz2 = y * z2
        let zz2 = z * z2
        
        return Vec3f(
            v.x * (1.0 - (yy2 + zz2)) + v.y * (xy2 - wz2) + v.z * (xz2 + wy2),
            v.x * (xy2 + wz2) + v.y * (1.0 - (xx2 + zz2)) + v.z * (yz2 - wx2),
            v.x * (xz2 - wy2) + v.y * (yz2 + wx2) + v.z * (1.0 - (xx2 + yy2))
        )
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