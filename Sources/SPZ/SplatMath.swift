import Foundation
import simd
import MetalPerformanceShaders

// Add MPSMatrix helper for batch operations
private let device = MTLCreateSystemDefaultDevice()!
private let commandQueue = device.makeCommandQueue()!

private func createMPSMatrix(from data: [Float], rows: Int, columns: Int) -> MPSMatrix? {
    let descriptor = MPSMatrixDescriptor(rows: rows,
                                       columns: columns,
                                       rowBytes: columns * MemoryLayout<Float>.stride,
                                       dataType: .float32)
    
    guard let matrix = device.makeBuffer(bytes: data,
                                       length: data.count * MemoryLayout<Float>.stride,
                                       options: .storageModeShared) else {
        return nil
    }
    
    return MPSMatrix(buffer: matrix, descriptor: descriptor)
}

// Add after createMPSMatrix function

private func batchMatrixOperation(
    inputs: [Float],
    inputRows: Int,
    inputCols: Int,
    operation: (MPSMatrix, MPSCommandBuffer) -> Void
) -> [Float]? {
    guard let inputMatrix = createMPSMatrix(from: inputs, rows: inputRows, columns: inputCols),
          let commandBuffer = commandQueue.makeCommandBuffer() as? MPSCommandBuffer else {
        return nil
    }
    
    operation(inputMatrix, commandBuffer)
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let result = inputMatrix.data.contents().assumingMemoryBound(to: Float.self)
    return Array(UnsafeBufferPointer(start: result, count: inputs.count))
}

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
    
    /// Converts quaternion to 3x3 rotation matrix
    var rotationMatrix: [Float] {
        let x = self.x
        let y = self.y
        let z = self.z
        let w = self.w
        
        let xx = x * x
        let xy = x * y
        let xz = x * z
        let xw = x * w
        let yy = y * y
        let yz = y * z
        let yw = y * w
        let zz = z * z
        let zw = z * w
        
        return [
            1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw),
            2 * (xy + zw),     1 - 2 * (xx + zz),     2 * (yz - xw),
            2 * (xz - yw),         2 * (yz + xw), 1 - 2 * (xx + yy)
        ]
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

extension GaussianCloud {
    /// Batch processes rotations using MPS
    func batchRotateVectors(_ vectors: [Vec3f]) -> [Vec3f] {
        guard vectors.count > 0 else { return [] }
        
        // Create rotation matrices from quaternions
        var rotationMatrices = [Float](repeating: 0, count: numPoints * 9)
        for i in 0..<numPoints {
            let q = Quat4f(rotations[i * 4..<i * 4 + 4])
            let mat3 = q.rotationMatrix
            for j in 0..<9 {
                rotationMatrices[i * 9 + j] = mat3[j]
            }
        }
        
        // Create MPS matrices
        guard let matrixR = createMPSMatrix(from: rotationMatrices, rows: numPoints, columns: 9),
              let matrixV = createMPSMatrix(from: vectors.flatMap { [$0.x, $0.y, $0.z] },
                                          rows: vectors.count,
                                          columns: 3),
              let resultBuffer = device.makeBuffer(length: vectors.count * 3 * MemoryLayout<Float>.stride,
                                                 options: .storageModeShared) else {
            return vectors
        }
        
        let resultDescriptor = MPSMatrixDescriptor(rows: vectors.count,
                                                 columns: 3,
                                                 rowBytes: 3 * MemoryLayout<Float>.stride,
                                                 dataType: .float32)
        let resultMatrix = MPSMatrix(buffer: resultBuffer, descriptor: resultDescriptor)
        
        // Create matrix multiplication kernel with alpha=1.0 and beta=0.0
        let matMul = MPSMatrixMultiplication(device: device,
                                           transposeLeft: false,
                                           transposeRight: true,
                                           resultRows: vectors.count,
                                           resultColumns: 3,
                                           interiorColumns: 3,
                                           alpha: 1.0,
                                           beta: 0.0)
        
        // Execute multiplication
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return vectors
        }
        
        matMul.encode(commandBuffer: commandBuffer,
                     leftMatrix: matrixV,
                     rightMatrix: matrixR,
                     resultMatrix: resultMatrix)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Convert result back to vectors
        let result = resultBuffer.contents().assumingMemoryBound(to: Float.self)
        var rotatedVectors = [Vec3f](repeating: .zero, count: vectors.count)
        for i in 0..<vectors.count {
            rotatedVectors[i] = Vec3f(result[i * 3],
                                    result[i * 3 + 1],
                                    result[i * 3 + 2])
        }
        
        return rotatedVectors
    }
}

// MARK: - Coordinate System Conversion

/// Returns true if the axes match between two coordinate systems
private func axesMatch(_ a: CoordinateSystem, _ b: CoordinateSystem) -> (Bool, Bool, Bool) {
    if a == .unspecified || b == .unspecified {
        return (true, true, true)
    }
    
    let aNum = a.rawValue - 1
    let bNum = b.rawValue - 1
    
    return (
        ((aNum >> 0) & 1) == ((bNum >> 0) & 1),
        ((aNum >> 1) & 1) == ((bNum >> 1) & 1),
        ((aNum >> 2) & 1) == ((bNum >> 2) & 1)
    )
}

/// Creates a coordinate converter between two coordinate systems
public func coordinateConverter(from: CoordinateSystem, to: CoordinateSystem) -> CoordinateConverter {
    let (xMatch, yMatch, zMatch) = axesMatch(from, to)
    let x: Float = xMatch ? 1.0 : -1.0
    let y: Float = yMatch ? 1.0 : -1.0
    let z: Float = zMatch ? 1.0 : -1.0
    
    var converter = CoordinateConverter()
    converter.flipP = Vec3f(x, y, z)
    converter.flipQ = Vec3f(y * z, x * z, x * y)
    
    // Flips for the 15 spherical harmonics coefficients
    converter.flipSh = [
        y,          // 0
        z,          // 1
        x,          // 2
        x * y,      // 3
        y * z,      // 4
        1.0,        // 5
        x * z,      // 6
        1.0,        // 7
        y,          // 8
        x * y * z,  // 9
        y,          // 10
        z,          // 11
        x,          // 12
        z,          // 13
        x           // 14
    ]
    
    return converter
}