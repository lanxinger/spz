import Foundation
import MetalPerformanceShaders
import simd

// Import required dependencies
import struct simd.simd_float3
import struct simd.simd_float4
import func simd.simd_length
import func simd.simd_dot

// No need to import SplatTypes as it's in the same module

// Add shared MPS resources at file scope
private let device = MTLCreateSystemDefaultDevice()!
private let commandQueue = device.makeCommandQueue()!

// Add helper function at file scope
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

// Simplify our approach - don't use MPSMatrixScale since it's causing problems
private func scaleMatrix(_ data: [Float], scale: Float, offset: Float = 0.0) -> [Float] {
    var result = [Float](repeating: 0, count: data.count)
    for i in 0..<data.count {
        result[i] = data[i] * scale + offset
    }
    return result
}

/// Represents a single inflated gaussian. Each gaussian has 236 bytes. Although the data is easier
/// to interpret in this format, it is not more precise than the packed format, since it was inflated.
public struct UnpackedGaussian {
    var position: Vec3f
    var rotation: Quat4f
    var scale: Vec3f
    var color: Vec3f
    var alpha: Float
    var shR: [Float]
    var shG: [Float]
    var shB: [Float]
    
    init() {
        position = .zero
        rotation = .zero
        scale = .zero
        color = .zero
        alpha = 0
        shR = Array(repeating: 0, count: 15)
        shG = Array(repeating: 0, count: 15)
        shB = Array(repeating: 0, count: 15)
    }
}

/// Represents a single low precision gaussian. Each gaussian has exactly 65 bytes, even if it does
/// not have full spherical harmonics.
public struct PackedGaussian {
    var position: [UInt8]
    var rotation: [UInt8]
    var scale: [UInt8]
    var color: [UInt8]
    var alpha: UInt8
    var shR: [UInt8]
    var shG: [UInt8]
    var shB: [UInt8]
    
    init() {
        position = Array(repeating: 0, count: 9)
        rotation = Array(repeating: 0, count: 4)
        scale = Array(repeating: 0, count: 3)
        color = Array(repeating: 0, count: 3)
        alpha = 0
        shR = Array(repeating: 0, count: 15)
        shG = Array(repeating: 0, count: 15)
        shB = Array(repeating: 0, count: 15)
    }
    
    func unpack(usesFloat16: Bool, usesQuaternionSmallestThree: Bool, fractionalBits: Int, converter: CoordinateConverter? = nil) -> UnpackedGaussian {
        var result = UnpackedGaussian()
        let c = converter ?? CoordinateConverter()
        
        // Unpack position based on format
        if usesFloat16 {
            guard position.count >= 6 else { return result }
            
            position.withUnsafeBytes { ptr in
                let halfPtr = ptr.baseAddress!.assumingMemoryBound(to: UInt16.self)
                for i in 0..<3 {
                    result.position[i] = c.flipP[i] * float16ToFloat32(halfPtr[i])
                }
            }
        } else {
            guard position.count >= 9 else { return result }
            
            // Decode 24-bit fixed point coordinates
            let scale = 1.0 / Float(1 << fractionalBits)
            for i in 0..<3 {
                var fixed32: Int32 = Int32(position[i * 3])
                fixed32 |= Int32(position[i * 3 + 1]) << 8
                fixed32 |= Int32(position[i * 3 + 2]) << 16
                if (fixed32 & 0x800000) != 0 {
                    fixed32 |= Int32(bitPattern: 0xFF000000)  // Sign extension
                }
                result.position[i] = c.flipP[i] * (Float(fixed32) * scale)
            }
        }
        
        // Unpack scale
        guard scale.count >= 3 else { return result }
        for i in 0..<3 {
            result.scale[i] = Float(scale[i]) / 16.0 - 10.0
        }
        
        // Unpack rotation
        if usesQuaternionSmallestThree {
            guard rotation.count >= 4 else { return result }
            unpackQuaternionSmallestThree(&result.rotation, rotation, c)
        } else {
            guard rotation.count >= 3 else { return result }
            unpackQuaternionFirstThree(&result.rotation, rotation, c)
        }
        
        // Unpack alpha
        result.alpha = invSigmoid(Float(alpha) / 255.0)
        
        // Unpack color
        guard color.count >= 3 else { return result }
        for i in 0..<3 {
            result.color[i] = (Float(color[i]) / 255.0 - 0.5) / colorScale
        }
        
        // Unpack SH coefficients
        let shCount = min(15, shR.count)
        for i in 0..<shCount {
            guard i < c.flipSh.count else { break }
            result.shR[i] = c.flipSh[i] * unquantizeSH(shR[i])
            result.shG[i] = c.flipSh[i] * unquantizeSH(shG[i])
            result.shB[i] = c.flipSh[i] * unquantizeSH(shB[i])
        }
        
        return result
    }
}

/// Represents a full splat with lower precision. Each splat has at most 64 bytes, although splats
/// with fewer spherical harmonics degrees will have less. The data is stored non-interleaved.
public struct PackedGaussians {
    public var numPoints: Int = 0
    public var shDegree: Int = 0
    public var fractionalBits: Int = 0
    public var antialiased: Bool = false
    public var usesQuaternionSmallestThree: Bool = true
    
    public var positions: [UInt8] = []
    public var scales: [UInt8] = []
    public var rotations: [UInt8] = []
    public var alphas: [UInt8] = []
    public var colors: [UInt8] = []
    public var sh: [UInt8] = []
    
    public var usesFloat16: Bool {
        positions.count == numPoints * 3 * 2
    }
    
    func at(_ index: Int) -> PackedGaussian {
        var result = PackedGaussian()
        let positionBits = usesFloat16 ? 6 : 9
        let start3 = index * 3
        let posStart = index * positionBits
        
        // Verify index is in bounds
        guard index >= 0 && index < numPoints else {
            return result
        }
        
        // Check bounds for position data
        guard posStart + positionBits <= positions.count else {
            return result
        }
        
        // Copy position data safely
        result.position = Array(positions[posStart..<(posStart + positionBits)])
        
        // Check bounds for other components
        guard start3 + 3 <= scales.count,
              start3 + 3 <= rotations.count,
              start3 + 3 <= colors.count,
              index < alphas.count else {
            return result
        }
        
        // Copy other components safely
        result.scale = Array(scales[start3..<(start3 + 3)])
        let rotationBytes = usesQuaternionSmallestThree ? 4 : 3
        let rotStart = index * rotationBytes
        guard rotStart + rotationBytes <= rotations.count else { return result }
        result.rotation = Array(rotations[rotStart..<(rotStart + rotationBytes)])
        result.color = Array(colors[start3..<(start3 + 3)])
        result.alpha = alphas[index]
        
        // Copy spherical harmonics
        let shDim = dimForDegree(shDegree)
        let shStart = index * shDim * 3
        
        // Verify SH array bounds
        guard shStart + (shDim * 3) <= sh.count else {
            return result
        }
        
        // Copy SH data safely
        for j in 0..<shDim {
            let idx = shStart + j * 3
            guard idx + 2 < sh.count else { break }
            result.shR[j] = sh[idx]
            result.shG[j] = sh[idx + 1]
            result.shB[j] = sh[idx + 2]
        }
        
        // Fill remaining SH coefficients with neutral value
        for j in shDim..<15 {
            result.shR[j] = 128
            result.shG[j] = 128
            result.shB[j] = 128
        }
        
        return result
    }
    
    func unpack(_ index: Int, converter: CoordinateConverter? = nil) -> UnpackedGaussian {
        at(index).unpack(usesFloat16: usesFloat16, usesQuaternionSmallestThree: usesQuaternionSmallestThree, fractionalBits: fractionalBits, converter: converter)
    }
}

// MARK: - Float16 Support

/// Convert a 16-bit float (half precision) to 32-bit float
private func float16ToFloat32(_ half: UInt16) -> Float {
    let sign = (half >> 15) & 0x1
    let exponent = (half >> 10) & 0x1f
    let mantissa = half & 0x3ff
    
    let signMul: Float = sign == 1 ? -1.0 : 1.0
    
    if exponent == 0 {
        // Subnormal numbers (no exponent, 0 in the mantissa decimal)
        return signMul * Float(pow(2.0, -14.0)) * Float(mantissa) / 1024.0
    }
    
    if exponent == 31 {
        // Infinity or NaN
        return mantissa != 0 ? Float.nan : signMul * Float.infinity
    }
    
    // Non-zero exponent implies 1 in the mantissa decimal
    return signMul * Float(pow(2.0, Double(exponent) - 15.0)) * (1.0 + Float(mantissa) / 1024.0)
}

// Packed memory layout for better cache utilization
@frozen
public struct PackedPoint {
    var position: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)  // 9 bytes
    var rotation: (UInt8, UInt8, UInt8, UInt8)  // 4 bytes (for smallest-three encoding)
    var scale: (UInt8, UInt8, UInt8)     // 3 bytes
    var color: (UInt8, UInt8, UInt8)     // 3 bytes
    var alpha: UInt8                      // 1 byte
    // Total: 20 bytes, aligned to 4 bytes = 20 bytes
}

/// Unpacks quaternion using first-three encoding from 3 bytes  
private func unpackQuaternionFirstThree(_ result: inout Quat4f, _ rotation: [UInt8], _ c: CoordinateConverter = CoordinateConverter()) {
    guard rotation.count >= 3 else { return }
    
    let xyz = Vec3f(
        Float(rotation[0]),
        Float(rotation[1]),
        Float(rotation[2])
    ) / 127.5 - Vec3f(1, 1, 1)
    
    // Apply coordinate flips
    let flippedXyz = Vec3f(
        xyz.x * c.flipQ.x,
        xyz.y * c.flipQ.y,
        xyz.z * c.flipQ.z
    )
    
    result.x = flippedXyz.x
    result.y = flippedXyz.y
    result.z = flippedXyz.z
    // Compute the real component - we know the quaternion is normalized and w is non-negative
    result.w = sqrt(max(0.0, 1.0 - flippedXyz.squaredNorm))
}

/// Unpacks quaternion using smallest-three encoding from 4 bytes
private func unpackQuaternionSmallestThree(_ result: inout Quat4f, _ rotation: [UInt8], _ c: CoordinateConverter) {
    guard rotation.count >= 4 else { return }
    
    // Extract the largest component index (2 bits)
    let largestIdx = Int(rotation[3] >> 6)
    
    // Extract 10-bit signed values for the three smallest components
    var components = [Float](repeating: 0, count: 4)
    
    // First component: bits 0-9 from bytes 0-1
    var val1 = Int16(rotation[0]) | (Int16(rotation[1] & 0x03) << 8)
    if val1 >= 512 { val1 -= 1024 } // Sign extension
    
    // Second component: bits 2-11 from bytes 1-2  
    var val2 = Int16((rotation[1] >> 2) | ((rotation[2] & 0x0F) << 6))
    if val2 >= 512 { val2 -= 1024 } // Sign extension
    
    // Third component: bits 4-13 from bytes 2-3
    var val3 = Int16((rotation[2] >> 4) | ((rotation[3] & 0x3F) << 4))
    if val3 >= 512 { val3 -= 1024 } // Sign extension
    
    // Convert to normalized float values
    let vals = [Float(val1), Float(val2), Float(val3)]
    
    // Place the three smallest components
    var compIdx = 0
    for i in 0..<4 {
        if i != largestIdx {
            let normalizedVal = sqrt1_2 * Float(vals[compIdx]) / Float((1 << 9) - 1)
            components[i] = normalizedVal * c.flipQ[i]
            compIdx += 1
        }
    }
    
    // Compute the largest component using quaternion normalization
    let sumSquares = components[0] * components[0] + components[1] * components[1] + 
                    components[2] * components[2] + components[3] * components[3]
    components[largestIdx] = sqrt(max(0.0, 1.0 - sumSquares))
    
    result.x = components[0]
    result.y = components[1] 
    result.z = components[2]
    result.w = components[3]
}

private func unpackPositionsVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    var result = [Float](repeating: 0, count: count * 3)
    
    if packed.usesFloat16 {
        // Process 4 positions (12 components) at a time
        let simdCount = (count * 3) / 12 * 12
        for i in stride(from: 0, to: simdCount, by: 12) {
            let sourceIdx = i * 2
            let values = packed.positions.withUnsafeBytes { ptr -> SIMD4<Float> in
                let uint16Ptr = ptr.baseAddress!.assumingMemoryBound(to: UInt16.self)
                return SIMD4(
                    float16ToFloat32(uint16Ptr[sourceIdx/2]),
                    float16ToFloat32(uint16Ptr[sourceIdx/2 + 1]),
                    float16ToFloat32(uint16Ptr[sourceIdx/2 + 2]),
                    float16ToFloat32(uint16Ptr[sourceIdx/2 + 3])
                )
            }
            
            for j in 0..<4 {
                result[i/2 + j] = values[j]
            }
        }
    } else {
        let scale = 1.0 / Float(1 << packed.fractionalBits)
        // Process 4 positions (12 components) at a time
        let simdCount = (count * 3) / 12 * 12
        for i in stride(from: 0, to: simdCount, by: 12) {
            let sourceIdx = i * 3
            var fixed32 = SIMD4<Int32>()
            
            for j in 0..<4 {
                let idx = sourceIdx + j * 3
                fixed32[j] = Int32(packed.positions[idx]) |
                            (Int32(packed.positions[idx + 1]) << 8) |
                            (Int32(packed.positions[idx + 2]) << 16)
                if (fixed32[j] & 0x800000) != 0 {
                    fixed32[j] |= Int32(bitPattern: 0xFF000000)
                }
            }
            
            let floatValues = SIMD4<Float>(fixed32) * scale
            for j in 0..<4 {
                result[i/3 + j] = floatValues[j]
            }
        }
    }
    
    return result
}

private func prefetchNextChunk(_ data: UnsafePointer<UInt8>, offset: Int, size: Int) {
    #if arch(arm64)
    // On ARM64, we can use memory barriers instead of prefetch
    for i in stride(from: 0, to: size, by: 64) {
        // Memory barrier to ensure data is loaded into cache
        _ = data[offset + i]
    }
    #endif
}

private func unpackColorsVectorized(_ packed: PackedGaussians) -> [Float] {
    // Convert UInt8 to Float
    let inputs = packed.colors.map { Float($0) }
    
    // Apply scaling and offset directly
    return scaleMatrix(inputs, scale: 1.0 / (255.0 * colorScale), offset: -0.5 / colorScale)
}

private func unpackScalesVectorized(_ packed: PackedGaussians) -> [Float] {
    // Convert UInt8 to Float
    let inputs = packed.scales.map { Float($0) }
    
    // Apply scaling and offset directly
    return scaleMatrix(inputs, scale: 1.0 / 16.0, offset: -10.0)
}

private func unpackAlphasVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    
    // Convert to [0,1] range
    let inputs = packed.alphas.map { Float($0) / 255.0 }
    
    // Apply inverse sigmoid directly
    var result = [Float](repeating: 0, count: count)
    for i in 0..<count {
        let x = max(0.0001, min(0.9999, inputs[i]))
        result[i] = log(x / (1.0 - x))
    }
    return result
}

// We're using the module-wide helper functions and constants