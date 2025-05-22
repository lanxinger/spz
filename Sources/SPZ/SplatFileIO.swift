import Foundation
import MetalPerformanceShaders

// MARK: - MPS Setup
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

// Import zlib types and constants
private let ZLIB_VERSION = "1.2.11"
private let MAX_WBITS: Int32 = 15
private let Z_OK: Int32 = 0
private let Z_STREAM_END: Int32 = 1
private let Z_NO_FLUSH: Int32 = 0
private let Z_FINISH: Int32 = 4
private let Z_DEFAULT_COMPRESSION: Int32 = -1
private let Z_DEFLATED: Int32 = 8
private let Z_DEFAULT_STRATEGY: Int32 = 0

private struct z_stream {
    var next_in: UnsafeMutablePointer<UInt8>?
    var avail_in: UInt32
    var total_in: UInt
    var next_out: UnsafeMutablePointer<UInt8>?
    var avail_out: UInt32
    var total_out: UInt
    var msg: UnsafePointer<Int8>?
    var state: OpaquePointer?
    var zalloc: OpaquePointer?
    var zfree: OpaquePointer?
    var opaque: OpaquePointer?
    var data_type: Int32
    var adler: UInt
    var reserved: UInt
}

@_silgen_name("deflateInit2_")
private func deflateInit2_(_ strm: UnsafeMutablePointer<z_stream>,
                          _ level: Int32,
                          _ method: Int32,
                          _ windowBits: Int32,
                          _ memLevel: Int32,
                          _ strategy: Int32,
                          _ version: UnsafePointer<Int8>,
                          _ stream_size: Int32) -> Int32

@_silgen_name("deflate")
private func deflate(_ strm: UnsafeMutablePointer<z_stream>, _ flush: Int32) -> Int32

@_silgen_name("deflateEnd")
private func deflateEnd(_ strm: UnsafeMutablePointer<z_stream>) -> Int32

@_silgen_name("crc32")
private func crc32(_ crc: UInt32, _ buf: UnsafePointer<UInt8>?, _ len: UInt32) -> UInt32

// Helper extension for UInt32 to Data conversion
private extension UInt32 {
    var data: Data {
        withUnsafeBytes(of: self.littleEndian) { Data($0) }
    }
}

/// Header structure for packed gaussians file format
private struct PackedGaussiansHeader {
    static let magic: UInt32 = 0x5053474e  // NGSP = Niantic gaussian splat
    static let version: UInt32 = 2
    
    var magic: UInt32 = PackedGaussiansHeader.magic
    var version: UInt32 = PackedGaussiansHeader.version
    var numPoints: UInt32 = 0
    var shDegree: UInt8 = 0
    var fractionalBits: UInt8 = 0
    var flags: UInt8 = 0
    var reserved: UInt8 = 0
    
    static let size = MemoryLayout<PackedGaussiansHeader>.size
    
    init() {}
    
    init(data: Data) throws {
        guard data.count >= PackedGaussiansHeader.size else {
            throw SPZError.invalidHeader
        }
        
        magic = data.withUnsafeBytes { $0.load(as: UInt32.self) }
        version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        numPoints = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) }
        shDegree = data[12]
        fractionalBits = data[13]
        flags = data[14]
        reserved = data[15]
    }
    
    func serialize() -> Data {
        var data = Data(capacity: PackedGaussiansHeader.size)
        data.append(contentsOf: withUnsafeBytes(of: magic) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: version) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: numPoints) { Data($0) })
        data.append(shDegree)
        data.append(fractionalBits)
        data.append(flags)
        data.append(reserved)
        return data
    }
}

extension PackedGaussians {
    
    /// Serializes the packed gaussians to binary format
    func serialize() -> Data {
        var header = PackedGaussiansHeader()
        header.numPoints = UInt32(numPoints)
        header.shDegree = UInt8(shDegree)
        header.fractionalBits = UInt8(fractionalBits)
        header.flags = antialiased ? 0x1 : 0x0
        
        var data = header.serialize()
        data.append(contentsOf: positions)
        data.append(contentsOf: alphas)
        data.append(contentsOf: colors)
        data.append(contentsOf: scales)
        data.append(contentsOf: rotations)
        data.append(contentsOf: sh)
        
        return data
    }
    
    /// Deserializes packed gaussians from binary data
    public static func deserialize(_ data: Data) throws -> PackedGaussians {
        let header = try PackedGaussiansHeader(data: data)
        
        guard header.magic == PackedGaussiansHeader.magic else {
            throw SPZError.invalidHeader
        }
        
        guard header.version >= 1 && header.version <= 2 else {
            throw SPZError.unsupportedVersion
        }
        
        let maxPointsToRead = 10_000_000
        guard header.numPoints <= maxPointsToRead else {
            throw SPZError.tooManyPoints
        }
        
        guard header.shDegree <= 3 else {
            throw SPZError.unsupportedSHDegree
        }
        
        let numPoints = Int(header.numPoints)
        let shDim = dimForDegree(Int(header.shDegree))
        let usesFloat16 = header.version == 1
        
        // Calculate expected sizes
        let posSize = numPoints * 3 * (usesFloat16 ? 2 : 3)
        let alphasSize = numPoints
        let colorsSize = numPoints * 3
        let scalesSize = numPoints * 3
        let rotationsSize = numPoints * 3
        let shSize = numPoints * shDim * 3
        
        // Calculate total expected size
        let expectedSize = PackedGaussiansHeader.size + posSize + alphasSize + colorsSize + 
                         scalesSize + rotationsSize + shSize
        
        guard data.count >= expectedSize else {
            print("[SPZ] Error: Data size mismatch. Expected at least \(expectedSize) bytes, got \(data.count)")
            throw SPZError.invalidData
        }
        
        var result = PackedGaussians()
        result.numPoints = numPoints
        result.shDegree = Int(header.shDegree)
        result.fractionalBits = Int(header.fractionalBits)
        result.antialiased = (header.flags & 0x1) != 0
        
        var offset = PackedGaussiansHeader.size
        
        // Safe array slicing helper
        func safeSlice(_ size: Int) throws -> [UInt8] {
            guard offset + size <= data.count else {
                print("[SPZ] Error: Trying to read \(size) bytes at offset \(offset), but data is only \(data.count) bytes")
                throw SPZError.invalidData
            }
            let result = Array(data[offset..<(offset + size)])
            offset += size
            return result
        }
        
        // Read components with bounds checking
        result.positions = try safeSlice(posSize)
        result.alphas = try safeSlice(alphasSize)
        result.colors = try safeSlice(colorsSize)
        result.scales = try safeSlice(scalesSize)
        result.rotations = try safeSlice(rotationsSize)
        result.sh = try safeSlice(shSize)
        
        // Verify final sizes
        guard checkSizes(result, numPoints: numPoints, shDim: shDim, usesFloat16: usesFloat16) else {
            print("[SPZ] Error: Component size verification failed")
            throw SPZError.invalidData
        }
        
        return result
    }
}

// MARK: - Public API

public extension GaussianCloud {
    
    /// Saves the gaussian cloud to a compressed SPZ file
    static func save(_ cloud: GaussianCloud, to url: URL) throws {
        let packed = packGaussians(cloud)
        let serialized = packed.serialize()
        
        // Create gzip header
        var compressedData = Data([0x1f, 0x8b])  // Gzip magic number
        compressedData.append(0x08)  // Compression method (deflate)
        compressedData.append(0x00)  // Flags
        compressedData.append(contentsOf: [0x00, 0x00, 0x00, 0x00])  // Time
        compressedData.append(0x00)  // Extra flags
        compressedData.append(0x00)  // OS (unknown)
        
        // Compress the data
        var stream = z_stream(
            next_in: nil,
            avail_in: 0,
            total_in: 0,
            next_out: nil,
            avail_out: 0,
            total_out: 0,
            msg: nil,
            state: nil,
            zalloc: nil,
            zfree: nil,
            opaque: nil,
            data_type: 0,
            adler: 0,
            reserved: 0
        )
        
        // Create a buffer that will live for the duration of compression
        let sourceBuffer = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: serialized.count)
        defer { sourceBuffer.deallocate() }
        
        _ = serialized.copyBytes(to: sourceBuffer)
        stream.next_in = sourceBuffer.baseAddress
        stream.avail_in = UInt32(serialized.count)
        
        let chunkSize = 16384
        var buffer = [UInt8](repeating: 0, count: chunkSize)
        
        let initResult = withUnsafeMutablePointer(to: &stream) { streamPtr in
            deflateInit2_(streamPtr,
                         Z_DEFAULT_COMPRESSION,
                         Z_DEFLATED,
                         -MAX_WBITS,  // Raw deflate
                         8,
                         Z_DEFAULT_STRATEGY,
                         ZLIB_VERSION,
                         Int32(MemoryLayout<z_stream>.size))
        }
        
        guard initResult == Z_OK else {
            throw SPZError.compressionError
        }
        
        defer {
            _ = withUnsafeMutablePointer(to: &stream) { streamPtr in
                deflateEnd(streamPtr)
            }
        }
        
        repeat {
            buffer.withUnsafeMutableBufferPointer { bufferPtr in
                stream.next_out = bufferPtr.baseAddress
                stream.avail_out = UInt32(chunkSize)
            }
            
            let deflateResult = withUnsafeMutablePointer(to: &stream) { streamPtr in
                deflate(streamPtr, Z_FINISH)
            }
            
            guard deflateResult >= Z_OK else {
                throw SPZError.compressionError
            }
            
            let bytesCompressed = chunkSize - Int(stream.avail_out)
            compressedData.append(contentsOf: buffer.prefix(bytesCompressed))
            
        } while stream.avail_out == 0
        
        // Add CRC32 and size at the end (gzip footer)
        var crc: UInt32 = 0
        crc = crc32(0, nil, 0)
        crc = sourceBuffer.withUnsafeBytes { bufferPtr in
            crc32(crc, bufferPtr.baseAddress?.assumingMemoryBound(to: UInt8.self), UInt32(serialized.count))
        }
        compressedData.append(contentsOf: withUnsafeBytes(of: crc.littleEndian) { Data($0) })
        compressedData.append(contentsOf: withUnsafeBytes(of: UInt32(serialized.count).littleEndian) { Data($0) })
        
        try compressedData.write(to: url)
    }
    
    /// Loads a gaussian cloud from a compressed SPZ file
    static func load(from url: URL) throws -> GaussianCloud {
        let data = try Data(contentsOf: url)
        guard let decompressed = decompressGzipped(data) else {
            throw SPZError.decompressionError
        }
        let packed = try PackedGaussians.deserialize(decompressed)
        return unpackGaussians(packed)
    }
}

// MARK: - Private Helpers

public func unpackGaussians(_ packed: PackedGaussians) -> GaussianCloud {
    let numPoints = packed.numPoints
    let shDim = dimForDegree(packed.shDegree)
    let usesFloat16 = packed.usesFloat16
    
    guard checkSizes(packed, numPoints: numPoints, shDim: shDim, usesFloat16: usesFloat16) else {
        print("[SPZ] Error: Size check failed")
        return GaussianCloud()
    }
    
    var result = GaussianCloud()
    result.numPoints = numPoints
    result.shDegree = packed.shDegree
    result.antialiased = packed.antialiased
    
    // Pre-allocate arrays
    result.positions = Array(repeating: 0, count: numPoints * 3)
    result.scales = Array(repeating: 0, count: numPoints * 3)
    result.rotations = Array(repeating: 0, count: numPoints * 4)
    result.alphas = Array(repeating: 0, count: numPoints)
    result.colors = Array(repeating: 0, count: numPoints * 3)
    result.sh = Array(repeating: 0, count: numPoints * shDim * 3)
    
    print("[SPZ] Unpacking \(numPoints) points with SH degree \(packed.shDegree)")
    
    // Progress tracking
    let progressInterval = max(1, numPoints / 100)  // Update every 1%
    var lastProgress = 0
    
    if usesFloat16 {
        // Decode legacy float16 format
        for i in 0..<numPoints {
            // Update progress
            let progress = (i * 100) / numPoints
            if progress != lastProgress && i % progressInterval == 0 {
                print("\r[SPZ] Unpacking progress: \(progress)%", terminator: "")
                fflush(stdout)
                lastProgress = progress
            }
            
            let baseIdx = i * 6  // 2 bytes per component, 3 components
            guard baseIdx + 5 < packed.positions.count else {
                print("[SPZ] Error: Position data out of bounds at point \(i)")
                return result
            }
            
            let x = UInt16(packed.positions[baseIdx]) | (UInt16(packed.positions[baseIdx + 1]) << 8)
            let y = UInt16(packed.positions[baseIdx + 2]) | (UInt16(packed.positions[baseIdx + 3]) << 8)
            let z = UInt16(packed.positions[baseIdx + 4]) | (UInt16(packed.positions[baseIdx + 5]) << 8)
            
            result.positions[i * 3 + 0] = float16ToFloat32(x)
            result.positions[i * 3 + 1] = float16ToFloat32(y)
            result.positions[i * 3 + 2] = float16ToFloat32(z)
        }
    } else {
        // Decode 24-bit fixed point coordinates
        let scale = 1.0 / Float(1 << packed.fractionalBits)
        for i in 0..<numPoints {
            // Update progress
            let progress = (i * 100) / numPoints
            if progress != lastProgress && i % progressInterval == 0 {
                print("\r[SPZ] Unpacking progress: \(progress)%", terminator: "")
                fflush(stdout)
                lastProgress = progress
            }
            
            let baseIdx = i * 9  // 3 bytes per component, 3 components
            guard baseIdx + 8 < packed.positions.count else {
                print("[SPZ] Error: Position data out of bounds at point \(i)")
                return result
            }
            
            for j in 0..<3 {
                let idx = baseIdx + j * 3
                var fixed32: Int32 = Int32(packed.positions[idx])
                fixed32 |= Int32(packed.positions[idx + 1]) << 8
                fixed32 |= Int32(packed.positions[idx + 2]) << 16
                if (fixed32 & 0x800000) != 0 {
                    fixed32 |= Int32(bitPattern: 0xFF000000)
                }
                result.positions[i * 3 + j] = Float(fixed32) * scale
            }
        }
    }
    
    // Unpack scales
    for i in 0..<numPoints {
        guard i * 3 + 2 < packed.scales.count else {
            print("[SPZ] Error: Scale data out of bounds at point \(i)")
            return result
        }
        for j in 0..<3 {
            result.scales[i * 3 + j] = Float(packed.scales[i * 3 + j]) / 16.0 - 10.0
        }
    }
    
    // Unpack rotations
    for i in 0..<numPoints {
        guard i * 3 + 2 < packed.rotations.count else {
            print("[SPZ] Error: Rotation data out of bounds at point \(i)")
            return result
        }
        let xyz = Vec3f(
            Float(packed.rotations[i * 3 + 0]),
            Float(packed.rotations[i * 3 + 1]),
            Float(packed.rotations[i * 3 + 2])
        ) / 127.5 - Vec3f(1, 1, 1)
        
        result.rotations[i * 4 + 0] = xyz.x
        result.rotations[i * 4 + 1] = xyz.y
        result.rotations[i * 4 + 2] = xyz.z
        result.rotations[i * 4 + 3] = sqrt(max(0.0, 1.0 - xyz.squaredNorm))
    }
    
    // Unpack alphas
    for i in 0..<numPoints {
        guard i < packed.alphas.count else {
            print("[SPZ] Error: Alpha data out of bounds at point \(i)")
            return result
        }
        result.alphas[i] = invSigmoid(Float(packed.alphas[i]) / 255.0)
    }
    
    // Unpack colors
    for i in 0..<numPoints {
        guard i * 3 + 2 < packed.colors.count else {
            print("[SPZ] Error: Color data out of bounds at point \(i)")
            return result
        }
        for j in 0..<3 {
            result.colors[i * 3 + j] = (Float(packed.colors[i * 3 + j]) / 255.0 - 0.5) / colorScale
        }
    }
    
    // Unpack spherical harmonics
    let shStride = shDim * 3
    for i in 0..<numPoints {
        guard i * shStride + shStride - 1 < packed.sh.count else {
            print("[SPZ] Error: SH data out of bounds at point \(i)")
            return result
        }
        for j in 0..<shStride {
            result.sh[i * shStride + j] = unquantizeSH(packed.sh[i * shStride + j])
        }
    }
    
    print("\r[SPZ] Unpacking complete: \(numPoints) points")
    return result
}

private func packGaussians(_ cloud: GaussianCloud) -> PackedGaussians {
    guard checkSizes(cloud) else {
        return PackedGaussians()
    }
    
    let numPoints = cloud.numPoints
    let shDim = dimForDegree(cloud.shDegree)
    
    // Use 12 bits for the fractional part of coordinates (~0.25 millimeter resolution)
    var packed = PackedGaussians()
    packed.numPoints = numPoints
    packed.shDegree = cloud.shDegree
    packed.fractionalBits = 12
    packed.antialiased = cloud.antialiased
    
    // Allocate arrays
    packed.positions = Array(repeating: 0, count: numPoints * 3 * 3)
    packed.scales = Array(repeating: 0, count: numPoints * 3)
    packed.rotations = Array(repeating: 0, count: numPoints * 3)
    packed.alphas = Array(repeating: 0, count: numPoints)
    packed.colors = Array(repeating: 0, count: numPoints * 3)
    packed.sh = Array(repeating: 0, count: numPoints * shDim * 3)
    
    // Store coordinates as 24-bit fixed point values
    let scale = Float(1 << packed.fractionalBits)
    for i in 0..<numPoints * 3 {
        let fixed32 = Int32(round(cloud.positions[i] * scale))
        packed.positions[i * 3 + 0] = UInt8(fixed32 & 0xff)
        packed.positions[i * 3 + 1] = UInt8((fixed32 >> 8) & 0xff)
        packed.positions[i * 3 + 2] = UInt8((fixed32 >> 16) & 0xff)
    }
    
    // Pack scales
    for i in 0..<numPoints * 3 {
        packed.scales[i] = toUInt8((cloud.scales[i] + 10.0) * 16.0)
    }
    
    // Pack rotations
    for i in 0..<numPoints {
        // Normalize the quaternion, make w positive, then store xyz
        var q = Quat4f(
            cloud.rotations[i * 4 + 0],
            cloud.rotations[i * 4 + 1],
            cloud.rotations[i * 4 + 2],
            cloud.rotations[i * 4 + 3]
        ).normalized
        
        // Make w positive
        let scale: Float = q.w < 0 ? -127.5 : 127.5
        q = q * scale + Quat4f(repeating: 127.5)
        
        packed.rotations[i * 3 + 0] = toUInt8(q.x)
        packed.rotations[i * 3 + 1] = toUInt8(q.y)
        packed.rotations[i * 3 + 2] = toUInt8(q.z)
    }
    
    // Pack alphas
    for i in 0..<numPoints {
        packed.alphas[i] = toUInt8(sigmoid(cloud.alphas[i]) * 255.0)
    }
    
    // Pack colors
    for i in 0..<numPoints * 3 {
        packed.colors[i] = toUInt8(cloud.colors[i] * (colorScale * 255.0) + (0.5 * 255.0))
    }
    
    // Pack spherical harmonics
    if cloud.shDegree > 0 {
        // Spherical harmonics quantization parameters
        let sh1Bits = 5
        let shRestBits = 4
        let shPerPoint = dimForDegree(cloud.shDegree) * 3
        
        for i in stride(from: 0, to: numPoints * shPerPoint, by: shPerPoint) {
            var j = 0
            // There are 9 coefficients for degree 1
            while j < 9 {
                packed.sh[i + j] = quantizeSH(cloud.sh[i + j], bucketSize: 1 << (8 - sh1Bits))
                j += 1
            }
            while j < shPerPoint {
                packed.sh[i + j] = quantizeSH(cloud.sh[i + j], bucketSize: 1 << (8 - shRestBits))
                j += 1
            }
        }
    }
    
    return packed
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

// MARK: - Size Checking Helpers

private func checkSizes(_ cloud: GaussianCloud) -> Bool {
    guard cloud.numPoints >= 0 else { return false }
    guard cloud.shDegree >= 0 && cloud.shDegree <= 3 else { return false }
    guard cloud.positions.count == cloud.numPoints * 3 else { return false }
    guard cloud.scales.count == cloud.numPoints * 3 else { return false }
    guard cloud.rotations.count == cloud.numPoints * 4 else { return false }
    guard cloud.alphas.count == cloud.numPoints else { return false }
    guard cloud.colors.count == cloud.numPoints * 3 else { return false }
    guard cloud.sh.count == cloud.numPoints * dimForDegree(cloud.shDegree) * 3 else { return false }
    return true
}

private func checkSizes(_ packed: PackedGaussians, numPoints: Int, shDim: Int, usesFloat16: Bool) -> Bool {
    guard packed.positions.count == numPoints * 3 * (usesFloat16 ? 2 : 3) else { return false }
    guard packed.scales.count == numPoints * 3 else { return false }
    guard packed.rotations.count == numPoints * 3 else { return false }
    guard packed.alphas.count == numPoints else { return false }
    guard packed.colors.count == numPoints * 3 else { return false }
    guard packed.sh.count == numPoints * shDim * 3 else { return false }
    return true
}

// Use block copy operations for arrays
private func copyArrays(from source: PackedGaussians, to destination: inout GaussianCloud) {
    let count = source.numPoints
    
    // Copy alphas using unsafe buffers
    source.alphas.withUnsafeBufferPointer { srcBuf in
        destination.alphas.withUnsafeMutableBufferPointer { dstBuf in
            guard let srcPtr = srcBuf.baseAddress,
                  let dstPtr = dstBuf.baseAddress else { return }
            // Convert UInt8 to Float while copying
            for i in 0..<count {
                dstPtr[i] = Float(srcPtr[i])
            }
        }
    }
    
    // Copy positions
    source.positions.withUnsafeBufferPointer { srcBuf in
        destination.positions.withUnsafeMutableBufferPointer { dstBuf in
            guard let srcPtr = srcBuf.baseAddress,
                  let dstPtr = dstBuf.baseAddress else { return }
            // Handle position conversion based on format
            if source.usesFloat16 {
                for i in 0..<count {
                    let baseIn = i * 6  // 2 bytes per component
                    let baseOut = i * 3
                    for j in 0..<3 {
                        let halfValue = UInt16(srcPtr[baseIn + j * 2]) |
                                      (UInt16(srcPtr[baseIn + j * 2 + 1]) << 8)
                        dstPtr[baseOut + j] = float16ToFloat32(halfValue)
                    }
                }
            } else {
                let scale = 1.0 / Float(1 << source.fractionalBits)
                for i in 0..<count {
                    let baseIn = i * 9  // 3 bytes per component
                    let baseOut = i * 3
                    for j in 0..<3 {
                        var fixed32: Int32 = Int32(srcPtr[baseIn + j * 3])
                        fixed32 |= Int32(srcPtr[baseIn + j * 3 + 1]) << 8
                        fixed32 |= Int32(srcPtr[baseIn + j * 3 + 2]) << 16
                        if (fixed32 & 0x800000) != 0 {
                            fixed32 |= Int32(bitPattern: 0xFF000000)
                        }
                        dstPtr[baseOut + j] = Float(fixed32) * scale
                    }
                }
            }
        }
    }
}

// Use DispatchQueue for parallel processing of large point sets
private func processPointsInParallel(packed: PackedGaussians, result: inout GaussianCloud) {
    let pointsPerThread = 1000
    let threadCount = (packed.numPoints + pointsPerThread - 1) / pointsPerThread
    
    // Create a temporary array to store results from each thread
    var threadResults = [(Range<Int>, [Float])](repeating: (0..<0, []), count: threadCount)
    
    DispatchQueue.concurrentPerform(iterations: threadCount) { threadIndex in
        let start = threadIndex * pointsPerThread
        let end = min(start + pointsPerThread, packed.numPoints)
        var localResult = [Float](repeating: 0, count: (end - start) * 3)
        
        // Process points in this range
        for i in start..<end {
            let localIndex = (i - start) * 3
            if packed.usesFloat16 {
                // Process float16 positions
                let baseIdx = i * 6
                for j in 0..<3 {
                    let halfValue = UInt16(packed.positions[baseIdx + j * 2]) |
                                  (UInt16(packed.positions[baseIdx + j * 2 + 1]) << 8)
                    localResult[localIndex + j] = float16ToFloat32(halfValue)
                }
            } else {
                // Process fixed-point positions
                let baseIdx = i * 9
                let scale = 1.0 / Float(1 << packed.fractionalBits)
                for j in 0..<3 {
                    var fixed32: Int32 = Int32(packed.positions[baseIdx + j * 3])
                    fixed32 |= Int32(packed.positions[baseIdx + j * 3 + 1]) << 8
                    fixed32 |= Int32(packed.positions[baseIdx + j * 3 + 2]) << 16
                    if (fixed32 & 0x800000) != 0 {
                        fixed32 |= Int32(bitPattern: 0xFF000000)
                    }
                    localResult[localIndex + j] = Float(fixed32) * scale
                }
            }
        }
        
        threadResults[threadIndex] = (start..<end, localResult)
    }
    
    // Combine results from all threads
    for (range, localResult) in threadResults {
        let destStart = range.startIndex * 3
        let count = range.count * 3
        result.positions.replaceSubrange(destStart..<(destStart + count), with: localResult)
    }
}

private func processSHBatch(_ cloud: GaussianCloud, startIdx: Int, count: Int) -> [UInt8] {
    let shDim = dimForDegree(cloud.shDegree)
    let sh1Bits = 5
    let shRestBits = 4
    var result = [UInt8](repeating: 0, count: count * shDim * 3)
    
    // Process 4 coefficients at a time using SIMD
    let simdCount = (count * shDim * 3) / 4 * 4
    let source = cloud.sh
    
    for i in stride(from: 0, to: simdCount, by: 4) {
        let values = SIMD4<Float>(
            source[startIdx + i],
            source[startIdx + i + 1],
            source[startIdx + i + 2],
            source[startIdx + i + 3]
        )
        
        let bucketSize: Float = i < (9 * count) ? Float(1 << (8 - sh1Bits)) : Float(1 << (8 - shRestBits))
        let bucketSizeVec = SIMD4<Float>(repeating: bucketSize)
        
        // Quantize values
        let quantized = values * 128.0 + SIMD4<Float>(repeating: 128.0)
        let rounded = quantized.rounded(.toNearestOrAwayFromZero)
        let halfBucket = bucketSizeVec * 0.5
        let bucketed = ((rounded + halfBucket) / bucketSizeVec).rounded(.towardZero) * bucketSizeVec
        
        // Manual clamping between 0 and 255
        let clamped = SIMD4<Float>(
            max(0, min(255, bucketed[0])),
            max(0, min(255, bucketed[1])),
            max(0, min(255, bucketed[2])),
            max(0, min(255, bucketed[3]))
        )
        
        for j in 0..<4 {
            result[i + j] = UInt8(clamped[j])
        }
    }
    
    // Handle remaining coefficients
    for i in simdCount..<(count * shDim * 3) {
        let bucketSize = i < (9 * count) ? 1 << (8 - sh1Bits) : 1 << (8 - shRestBits)
        result[i] = quantizeSH(source[startIdx + i], bucketSize: bucketSize)
    }
    
    return result
}

private func processPointsByCacheLine(_ packed: PackedGaussians, result: inout GaussianCloud) {
    // Process points in cache-line sized chunks (64 bytes typically)
    let pointsPerCacheLine = 64 / MemoryLayout<Float>.stride
    let count = packed.numPoints
    
    for baseIdx in stride(from: 0, to: count, by: pointsPerCacheLine) {
        let endIdx = min(baseIdx + pointsPerCacheLine, count)
        // Process this cache-line sized chunk of points
        for pointIdx in baseIdx..<endIdx {
            let posIdx = pointIdx * 3
            if packed.usesFloat16 {
                let baseIn = pointIdx * 6  // 2 bytes per component
                for j in 0..<3 {
                    let halfValue = UInt16(packed.positions[baseIn + j * 2]) |
                                  (UInt16(packed.positions[baseIn + j * 2 + 1]) << 8)
                    result.positions[posIdx + j] = float16ToFloat32(halfValue)
                }
            } else {
                let baseIn = pointIdx * 9  // 3 bytes per component
                let scale = 1.0 / Float(1 << packed.fractionalBits)
                for j in 0..<3 {
                    var fixed32: Int32 = Int32(packed.positions[baseIn + j * 3])
                    fixed32 |= Int32(packed.positions[baseIn + j * 3 + 1]) << 8
                    fixed32 |= Int32(packed.positions[baseIn + j * 3 + 2]) << 16
                    if (fixed32 & 0x800000) != 0 {
                        fixed32 |= Int32(bitPattern: 0xFF000000)
                    }
                    result.positions[posIdx + j] = Float(fixed32) * scale
                }
            }
        }
    }
}

private func processSHBatchMPS(_ cloud: GaussianCloud, startIdx: Int, count: Int) -> [UInt8] {
    // Fall back to CPU implementation if MPS is not available
    if #available(iOS 10.0, macOS 10.13, *) {
        let shDim = dimForDegree(cloud.shDegree)
        let sh1Bits = 5
        let shRestBits = 4
        
        // Create input matrix
        let source = Array(cloud.sh[startIdx..<(startIdx + count * shDim * 3)])
        guard let inputMatrix = createMPSMatrix(from: source,
                                              rows: count,
                                              columns: shDim * 3) else {
            return processSHBatch(cloud, startIdx: startIdx, count: count)
        }
        
        // Create scale and bias matrices
        let scaleData: [Float] = [128.0]
        let biasData: [Float] = [128.0]
        
        guard let scaleMatrix = createMPSMatrix(from: scaleData, rows: 1, columns: 1),
              let biasMatrix = createMPSMatrix(from: biasData, rows: 1, columns: 1) else {
            return processSHBatch(cloud, startIdx: startIdx, count: count)
        }
        
        // Create output buffer
        guard let outputBuffer = device.makeBuffer(length: count * shDim * 3 * MemoryLayout<Float>.stride,
                                                 options: .storageModeShared) else {
            return processSHBatch(cloud, startIdx: startIdx, count: count)
        }
        
        let outputDescriptor = MPSMatrixDescriptor(rows: count,
                                                 columns: shDim * 3,
                                                 rowBytes: shDim * 3 * MemoryLayout<Float>.stride,
                                                 dataType: .float32)
        let outputMatrix = MPSMatrix(buffer: outputBuffer, descriptor: outputDescriptor)
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return processSHBatch(cloud, startIdx: startIdx, count: count)
        }
        
        // Scale and bias
        let scaleKernel = MPSMatrixMultiplication(device: device,
                                                transposeLeft: false,
                                                transposeRight: false,
                                                resultRows: count,
                                                resultColumns: shDim * 3,
                                                interiorColumns: 1,
                                                alpha: 1.0,
                                                beta: 0.0)
        
        scaleKernel.encode(commandBuffer: commandBuffer,
                          leftMatrix: inputMatrix,
                          rightMatrix: scaleMatrix,
                          resultMatrix: outputMatrix)
        
        // Add bias using a matrix multiplication
        let biasKernel = MPSMatrixMultiplication(device: device,
                                               transposeLeft: false,
                                               transposeRight: false,
                                               resultRows: count,
                                               resultColumns: shDim * 3,
                                               interiorColumns: 1,
                                               alpha: 1.0,
                                               beta: 1.0)
        
        biasKernel.encode(commandBuffer: commandBuffer,
                         leftMatrix: outputMatrix,
                         rightMatrix: biasMatrix,
                         resultMatrix: outputMatrix)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Convert to UInt8 with bucketing
        let result = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        var quantized = [UInt8](repeating: 0, count: count * shDim * 3)
        
        for i in 0..<(count * shDim * 3) {
            let bucketSize = i < (9 * count) ? 1 << (8 - sh1Bits) : 1 << (8 - shRestBits)
            let value = result[i]
            let bucketed = ((value + Float(bucketSize) / 2) / Float(bucketSize)).rounded(.down) * Float(bucketSize)
            quantized[i] = UInt8(max(0, min(255, bucketed)))
        }
        
        return quantized
    } else {
        // Fall back to CPU implementation for older OS versions
        return processSHBatch(cloud, startIdx: startIdx, count: count)
    }
}
