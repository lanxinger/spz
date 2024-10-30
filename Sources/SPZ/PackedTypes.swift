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
        
        if usesFloat16 {
            // Decode legacy float16 format
            let halfData = position.withUnsafeBytes { ptr -> [UInt16] in
                Array(ptr.bindMemory(to: UInt16.self).prefix(3))
            }
            result.position = Vec3f(
                float16ToFloat32(halfData[0]),
                float16ToFloat32(halfData[1]),
                float16ToFloat32(halfData[2])
            )
        } else {
            // Decode 24-bit fixed point coordinates
            let scale = 1.0 / Float(1 << fractionalBits)
            for i in 0..<3 {
                var fixed32: Int32 = Int32(position[i * 3 + 0])
                fixed32 |= Int32(position[i * 3 + 1]) << 8
                fixed32 |= Int32(position[i * 3 + 2]) << 16
                // Sign extension - use Int32 bit pattern
                if (fixed32 & 0x800000) != 0 {
                    fixed32 |= Int32(bitPattern: 0xFF000000)
                }
                result.position[i] = Float(fixed32) * scale
            }
        }
        
        // Decode scales
        for i in 0..<3 {
            result.scale[i] = Float(scale[i]) / 16.0 - 10.0
        }
        
        // Decode rotation
        let xyz = Vec3f(
            Float(rotation[0]),
            Float(rotation[1]),
            Float(rotation[2])
        ) / 127.5 - Vec3f(1, 1, 1)
        
        result.rotation.x = xyz.x
        result.rotation.y = xyz.y
        result.rotation.z = xyz.z
        // Compute the real component - we know the quaternion is normalized and w is non-negative
        result.rotation.w = sqrt(max(0.0, 1.0 - xyz.squaredNorm))
        
        // Decode alpha
        result.alpha = invSigmoid(Float(alpha) / 255.0)
        
        // Decode colors
        for i in 0..<3 {
            result.color[i] = (Float(color[i]) / 255.0 - 0.5) / colorScale
        }
        
        // Decode spherical harmonics
        for i in 0..<15 {
            result.shR[i] = unquantizeSH(shR[i])
            result.shG[i] = unquantizeSH(shG[i])
            result.shB[i] = unquantizeSH(shB[i])
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