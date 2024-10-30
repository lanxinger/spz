import Foundation

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

/// Represents a single low precision gaussian. Each gaussian has exactly 64 bytes, even if it does
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
        rotation = Array(repeating: 0, count: 3)
        scale = Array(repeating: 0, count: 3)
        color = Array(repeating: 0, count: 3)
        alpha = 0
        shR = Array(repeating: 0, count: 15)
        shG = Array(repeating: 0, count: 15)
        shB = Array(repeating: 0, count: 15)
    }
    
    func unpack(usesFloat16: Bool, fractionalBits: Int) -> UnpackedGaussian {
        var result = UnpackedGaussian()
        
        // Use unsafe buffers for faster access
        position.withUnsafeBufferPointer { posPtr in
            rotation.withUnsafeBufferPointer { rotPtr in
                scale.withUnsafeBufferPointer { scalePtr in
                    // Unpack operations using direct pointer access
                    if usesFloat16 {
                        let halfPtr = posPtr.baseAddress!.withMemoryRebound(to: UInt16.self, capacity: 3) { $0 }
                        result.position = Vec3f(
                            float16ToFloat32(halfPtr[0]),
                            float16ToFloat32(halfPtr[1]),
                            float16ToFloat32(halfPtr[2])
                        )
                    }
                    // ... rest of unpacking logic
                }
            }
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
        print("[SPZ] Getting gaussian at index \(index)")
        var result = PackedGaussian()
        let positionBits = usesFloat16 ? 6 : 9
        let start3 = index * 3
        let posStart = index * positionBits
        
        // Verify index is in bounds
        guard index >= 0 && index < numPoints else {
            print("[SPZ] Error: Index \(index) out of bounds (numPoints: \(numPoints))")
            return result
        }
        
        print("[SPZ] Array sizes: positions=\(positions.count), scales=\(scales.count), rotations=\(rotations.count), colors=\(colors.count), alphas=\(alphas.count), sh=\(sh.count)")
        print("[SPZ] Accessing position data: start=\(posStart), bits=\(positionBits), total=\(positions.count)")
        
        // Verify array bounds for position data
        guard posStart + positionBits <= positions.count else {
            print("[SPZ] Error: Position data out of bounds at index \(index) (trying to access \(posStart + positionBits) but have \(positions.count) bytes)")
            return result
        }
        
        // Copy position data safely
        result.position = []
        for i in 0..<positionBits {
            result.position.append(positions[posStart + i])
        }
        
        print("[SPZ] Accessing component data: start=\(start3), total=\(scales.count)/\(rotations.count)/\(colors.count)")
        
        // Verify array bounds for other components
        guard start3 + 3 <= scales.count,
              start3 + 3 <= rotations.count,
              start3 + 3 <= colors.count,
              index < alphas.count else {
            print("[SPZ] Error: Component data out of bounds at index \(index)")
            print("[SPZ] Required: \(start3 + 3) bytes, have scales=\(scales.count), rotations=\(rotations.count), colors=\(colors.count)")
            return result
        }
        
        // Copy other components safely
        result.scale = []
        result.rotation = []
        result.color = []
        for i in 0..<3 {
            result.scale.append(scales[start3 + i])
            result.rotation.append(rotations[start3 + i])
            result.color.append(colors[start3 + i])
        }
        result.alpha = alphas[index]
        
        // Copy spherical harmonics
        let shDim = dimForDegree(shDegree)
        let shStart = index * shDim * 3
        
        print("[SPZ] Accessing SH data: start=\(shStart), dim=\(shDim), total=\(sh.count)")
        
        // Verify SH array bounds
        guard shStart + (shDim * 3) <= sh.count else {
            print("[SPZ] Error: SH data out of bounds at index \(index)")
            print("[SPZ] Required: \(shStart + (shDim * 3)) bytes, have \(sh.count)")
            return result
        }
        
        // Copy SH data safely
        for j in 0..<shDim {
            let idx = shStart + j * 3
            guard idx + 2 < sh.count else {
                print("[SPZ] Error: SH coefficient out of bounds at index \(idx)")
                break
            }
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
    
    func unpack(_ index: Int) -> UnpackedGaussian {
        at(index).unpack(usesFloat16: usesFloat16, fractionalBits: fractionalBits)
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
    var rotation: (UInt8, UInt8, UInt8)  // 3 bytes
    var scale: (UInt8, UInt8, UInt8)     // 3 bytes
    var color: (UInt8, UInt8, UInt8)     // 3 bytes
    var alpha: UInt8                      // 1 byte
    // Total: 19 bytes, aligned to 4 bytes = 20 bytes
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