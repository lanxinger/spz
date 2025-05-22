import Foundation
import MetalPerformanceShaders

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

// Add at the top of the file, after imports
private class MPSMatrixScale {
    let device: MTLDevice
    let alpha: Float
    let beta: Float
    let kernel: MPSMatrixUnaryKernel
    
    init(device: MTLDevice, alpha: Float, beta: Float = 0.0) {
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.kernel = MPSMatrixUnaryKernel(device: device)
    }
    
    func encode(commandBuffer: MPSCommandBuffer, inputMatrix: MPSMatrix, resultMatrix: MPSMatrix) {
        // Create a compute pipeline for the scale operation
        let functionConstants = MTLFunctionConstantValues()
        var alpha = self.alpha
        var beta = self.beta
        functionConstants.setConstantValue(&alpha, type: .float, index: 0)
        functionConstants.setConstantValue(&beta, type: .float, index: 1)
        
        let library = device.makeDefaultLibrary()
        guard let function = try? library?.makeFunction(name: "matrixScale", constantValues: functionConstants) else {
            return
        }
        
        guard let pipeline = try? device.makeComputePipelineState(function: function),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputMatrix.data, offset: 0, index: 0)
        encoder.setBuffer(resultMatrix.data, offset: 0, index: 1)
        encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 2)
        encoder.setBytes(&beta, length: MemoryLayout<Float>.size, index: 3)
        
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (inputMatrix.rows + 15) / 16,
            height: (inputMatrix.columns + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
    }
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

private func unpackColorsVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    
    // Create input matrix from packed colors
    let inputs = packed.colors.map { Float($0) }
    
    return batchMatrixOperation(
        inputs: inputs,
        inputRows: count,
        inputCols: 3
    ) { matrix, commandBuffer in
        let scale = MPSMatrixScale(device: device, 
                                 alpha: 1.0 / (255.0 * colorScale), 
                                 beta: -0.5 / colorScale)
        scale.encode(commandBuffer: commandBuffer, inputMatrix: matrix, resultMatrix: matrix)
    } ?? {
        // Fallback to CPU implementation
        var result = [Float](repeating: 0, count: count * 3)
        for i in 0..<(count * 3) {
            result[i] = (Float(packed.colors[i]) / 255.0 - 0.5) / colorScale
        }
        return result
    }()
}

private func unpackScalesVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    
    // Create input matrix from packed scales
    let inputs = packed.scales.map { Float($0) }
    
    return batchMatrixOperation(
        inputs: inputs,
        inputRows: count,
        inputCols: 3
    ) { matrix, commandBuffer in
        let scale = MPSMatrixScale(device: device, 
                                 alpha: 1.0 / 16.0, 
                                 beta: -10.0)
        scale.encode(commandBuffer: commandBuffer, inputMatrix: matrix, resultMatrix: matrix)
    } ?? {
        // Fallback to CPU implementation
        var result = [Float](repeating: 0, count: count * 3)
        for i in 0..<(count * 3) {
            result[i] = Float(packed.scales[i]) / 16.0 - 10.0
        }
        return result
    }()
}

private func unpackAlphasVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    
    // Create input matrix from packed alphas
    let inputs = packed.alphas.map { Float($0) }
    
    return batchMatrixOperation(
        inputs: inputs,
        inputRows: count,
        inputCols: 1
    ) { matrix, commandBuffer in
        // First scale to [0,1]
        let scale = MPSMatrixScale(device: device, alpha: 1.0 / 255.0)
        scale.encode(commandBuffer: commandBuffer, inputMatrix: matrix, resultMatrix: matrix)
        
        // Then apply inverse sigmoid using a compute pipeline
        let library = device.makeDefaultLibrary()
        let function = library?.makeFunction(name: "invSigmoid")
        let pipeline = try? device.makeComputePipelineState(function: function!)
        
        guard let pipeline = pipeline,
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(matrix.data, offset: 0, index: 0)
        
        let threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadGroups = (count + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(MTLSize(width: threadGroups, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()
    } ?? {
        // Fallback to CPU implementation
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = invSigmoid(Float(packed.alphas[i]) / 255.0)
        }
        return result
    }()
}

// Add the inverse sigmoid Metal shader:
#if os(macOS)
let invSigmoidShader = """
#include <metal_stdlib>
using namespace metal;

kernel void invSigmoid(
    device float* data [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    float x = data[id];
    data[id] = log(x / (1.0 - x));
}
"""
#endif