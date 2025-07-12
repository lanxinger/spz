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
    static let version: UInt32 = 3
    
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
        
        guard header.version >= 1 && header.version <= 3 else {
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
        let usesQuaternionSmallestThree = header.version >= 3
        
        // Calculate expected sizes
        let posSize = numPoints * 3 * (usesFloat16 ? 2 : 3)
        let alphasSize = numPoints
        let colorsSize = numPoints * 3
        let scalesSize = numPoints * 3
        let rotationsSize = numPoints * (usesQuaternionSmallestThree ? 4 : 3)
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
        result.usesQuaternionSmallestThree = usesQuaternionSmallestThree
        
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
    static func save(_ cloud: GaussianCloud, to url: URL, from coordinateSystem: CoordinateSystem = .unspecified) throws {
        let packed = packGaussians(cloud, from: coordinateSystem)
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
    static func load(from url: URL, to coordinateSystem: CoordinateSystem = .unspecified) throws -> GaussianCloud {
        let data = try Data(contentsOf: url)
        guard let decompressed = decompressGzipped(data) else {
            throw SPZError.decompressionError
        }
        let packed = try PackedGaussians.deserialize(decompressed)
        return unpackGaussians(packed, to: coordinateSystem)
    }
}

// MARK: - Private Helpers

public func unpackGaussians(_ packed: PackedGaussians, to coordinateSystem: CoordinateSystem = .unspecified) -> GaussianCloud {
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
    
    // Create coordinate converter if needed
    let converter = coordinateSystem != .unspecified ? 
        coordinateConverter(from: .rub, to: coordinateSystem) : nil
    
    // Process points with batch operations where possible
    // For positions, we use either a vectorized approach or a more cache-friendly one
    // depending on the data format
    
    if numPoints > 10000 {
        // For large point sets, use optimized unpacking
        if usesFloat16 {
            result.positions = unpackPositionsVectorized(packed)
        } else {
            processPointsByCacheLine(packed, result: &result)
        }
        
        // CPU fallbacks for the other components
        for i in 0..<numPoints {
            let i3 = i * 3
            
            // Scales
            for j in 0..<3 {
                result.scales[i3 + j] = Float(packed.scales[i3 + j]) / 16.0 - 10.0
            }
            
            // Colors
            for j in 0..<3 {
                result.colors[i3 + j] = (Float(packed.colors[i3 + j]) / 255.0 - 0.5) / colorScale
            }
            
            // Alpha
            result.alphas[i] = invSigmoid(Float(packed.alphas[i]) / 255.0)
            
            // Rotation
            let i4 = i * 4
            let xyz = Vec3f(
                Float(packed.rotations[i3 + 0]),
                Float(packed.rotations[i3 + 1]),
                Float(packed.rotations[i3 + 2])
            ) / 127.5 - Vec3f(1, 1, 1)
            
            result.rotations[i4 + 0] = xyz.x
            result.rotations[i4 + 1] = xyz.y
            result.rotations[i4 + 2] = xyz.z
            result.rotations[i4 + 3] = sqrt(max(0.0, 1.0 - xyz.squaredNorm))
            
            // SH
            let shStride = shDim * 3
            let shOffset = i * shStride
            for j in 0..<shStride {
                if j < packed.sh.count {
                    result.sh[shOffset + j] = unquantizeSH(packed.sh[shOffset + j])
                }
            }
        }
    } else {
        // For smaller point sets, use the standard approach
        let progressInterval = max(1, numPoints / 100)  // Update every 1%
        var lastProgress = 0
        
        for i in 0..<numPoints {
            // Update progress
            let progress = (i * 100) / numPoints
            if progress != lastProgress && i % progressInterval == 0 {
                print("\r[SPZ] Unpacking progress: \(progress)%", terminator: "")
                fflush(stdout)
                lastProgress = progress
            }
            
            // Unpack each gaussian using the at() method
            let unpacked = packed.unpack(i, converter: converter)
            
            // Copy data to result
            let i3 = i * 3
            let i4 = i * 4
            
            result.positions[i3..<(i3+3)] = [unpacked.position.x, unpacked.position.y, unpacked.position.z]
            result.scales[i3..<(i3+3)] = [unpacked.scale.x, unpacked.scale.y, unpacked.scale.z]
            result.rotations[i4..<(i4+4)] = [unpacked.rotation.x, unpacked.rotation.y, unpacked.rotation.z, unpacked.rotation.w]
            result.alphas[i] = unpacked.alpha
            result.colors[i3..<(i3+3)] = [unpacked.color.x, unpacked.color.y, unpacked.color.z]
            
            // Copy SH coefficients
            let shStride = shDim * 3
            let shOffset = i * shStride
            
            for j in 0..<shDim {
                result.sh[shOffset + j * 3 + 0] = unpacked.shR[j]
                result.sh[shOffset + j * 3 + 1] = unpacked.shG[j]
                result.sh[shOffset + j * 3 + 2] = unpacked.shB[j]
            }
        }
    }
    
    // If we used the optimized approach, apply coordinate transformation after the fact
    if numPoints > 10000 && coordinateSystem != .unspecified {
        result.convertCoordinates(from: .rub, to: coordinateSystem)
    }
    
    print("\r[SPZ] Unpacking complete: \(numPoints) points")
    return result
}

/// Packs quaternion using smallest-three encoding into 4 bytes
private func packQuaternionSmallestThree(_ q: Quat4f, _ rotations: inout [UInt8], _ offset: Int) {
    // Find the largest component by absolute value
    let absVals = [abs(q.x), abs(q.y), abs(q.z), abs(q.w)]
    let largestIdx = absVals.enumerated().max(by: { $0.element < $1.element })!.offset
    
    // Ensure the largest component is positive
    var quaternion = q
    if quaternion[largestIdx] < 0 {
        quaternion = -quaternion
    }
    
    // Get the three smallest components
    var smallestThree: [Float] = []
    for i in 0..<4 {
        if i != largestIdx {
            smallestThree.append(quaternion[i])
        }
    }
    
    // Convert to 10-bit signed integers
    let scale = Float(511.0)  // 2^9 - 1
    let vals = smallestThree.map { Int16(max(-511, min(511, $0 * scale))) }
    
    // Pack into 4 bytes
    // First 10 bits: vals[0]
    rotations[offset + 0] = UInt8(vals[0] & 0xFF)
    rotations[offset + 1] = UInt8(((vals[0] >> 8) & 0x03) | ((vals[1] & 0x3F) << 2))
    
    // Next 10 bits: vals[1] (bits 2-11)
    rotations[offset + 2] = UInt8(((vals[1] >> 6) & 0x0F) | ((vals[2] & 0x0F) << 4))
    
    // Last 10 bits: vals[2] (bits 4-13) + 2 bits for largest index
    rotations[offset + 3] = UInt8(((vals[2] >> 4) & 0x3F) | (UInt8(largestIdx) << 6))
}

private func packGaussians(_ cloud: GaussianCloud, from coordinateSystem: CoordinateSystem = .unspecified) -> PackedGaussians {
    guard checkSizes(cloud) else {
        return PackedGaussians()
    }
    
    let numPoints = cloud.numPoints
    let shDim = dimForDegree(cloud.shDegree)
    
    // Convert coordinate system if needed
    var cloudToConvert = cloud
    if coordinateSystem != .unspecified {
        cloudToConvert.convertCoordinates(from: coordinateSystem, to: .rub)
    }
    
    // Use 12 bits for the fractional part of coordinates (~0.25 millimeter resolution)
    var packed = PackedGaussians()
    packed.numPoints = numPoints
    packed.shDegree = cloud.shDegree
    packed.fractionalBits = 12
    packed.antialiased = cloud.antialiased
    packed.usesQuaternionSmallestThree = true  // Use version 3 encoding
    
    // Allocate arrays
    packed.positions = Array(repeating: 0, count: numPoints * 3 * 3)
    packed.scales = Array(repeating: 0, count: numPoints * 3)
    packed.rotations = Array(repeating: 0, count: numPoints * 4)  // 4 bytes for smallest-three encoding
    packed.alphas = Array(repeating: 0, count: numPoints)
    packed.colors = Array(repeating: 0, count: numPoints * 3)
    packed.sh = Array(repeating: 0, count: numPoints * shDim * 3)
    
    // Store coordinates as 24-bit fixed point values
    let scale = Float(1 << packed.fractionalBits)
    for i in 0..<numPoints * 3 {
        let posValue = cloudToConvert.positions[i]
        // Check for NaN or infinity values and replace with a safe value
        let safeValue = posValue.isFinite ? posValue : 0.0
        let fixed32 = Int32(round(safeValue * scale))
        packed.positions[i * 3 + 0] = UInt8(fixed32 & 0xff)
        packed.positions[i * 3 + 1] = UInt8((fixed32 >> 8) & 0xff)
        packed.positions[i * 3 + 2] = UInt8((fixed32 >> 16) & 0xff)
    }
    
    // Pack scales
    for i in 0..<numPoints * 3 {
        let scaleValue = cloudToConvert.scales[i]
        let safeValue = scaleValue.isFinite ? scaleValue : 0.0
        packed.scales[i] = toUInt8((safeValue + 10.0) * 16.0)
    }
    
    // Pack rotations
    for i in 0..<numPoints {
        // Normalize the quaternion, make w positive, then store xyz
        var q = Quat4f(
            cloudToConvert.rotations[i * 4 + 0].isFinite ? cloudToConvert.rotations[i * 4 + 0] : 0.0,
            cloudToConvert.rotations[i * 4 + 1].isFinite ? cloudToConvert.rotations[i * 4 + 1] : 0.0,
            cloudToConvert.rotations[i * 4 + 2].isFinite ? cloudToConvert.rotations[i * 4 + 2] : 0.0,
            cloudToConvert.rotations[i * 4 + 3].isFinite ? cloudToConvert.rotations[i * 4 + 3] : 1.0
        ).normalized
        
        // Pack quaternion using smallest-three encoding (version 3)
        packQuaternionSmallestThree(q, &packed.rotations, i * 4)
    }
    
    // Pack alphas
    for i in 0..<numPoints {
        let alphaValue = cloudToConvert.alphas[i]
        let safeValue = alphaValue.isFinite ? alphaValue : 0.0
        packed.alphas[i] = toUInt8(sigmoid(safeValue) * 255.0)
    }
    
    // Pack colors
    for i in 0..<numPoints * 3 {
        let colorValue = cloudToConvert.colors[i]
        let safeValue = colorValue.isFinite ? colorValue : 0.0
        packed.colors[i] = toUInt8(safeValue * (colorScale * 255.0) + (0.5 * 255.0))
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
                let shValue = cloudToConvert.sh[i + j]
                let safeValue = shValue.isFinite ? shValue : 0.0
                packed.sh[i + j] = quantizeSH(safeValue, bucketSize: 1 << (8 - sh1Bits))
                j += 1
            }
            while j < shPerPoint {
                let shValue = cloudToConvert.sh[i + j]
                let safeValue = shValue.isFinite ? shValue : 0.0
                packed.sh[i + j] = quantizeSH(safeValue, bucketSize: 1 << (8 - shRestBits))
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
    guard packed.rotations.count == numPoints * 4 else { return false }  // 4 bytes for smallest-three encoding
    guard packed.alphas.count == numPoints else { return false }
    guard packed.colors.count == numPoints * 3 else { return false }
    guard packed.sh.count == numPoints * shDim * 3 else { return false }
    return true
}

// MARK: - Vectorized Unpacking Functions

/// Unpacks positions using vectorized operations for better performance
private func unpackPositionsVectorized(_ packed: PackedGaussians) -> [Float] {
    let count = packed.numPoints
    var result = [Float](repeating: 0, count: count * 3)
    
    if packed.usesFloat16 {
        // Process positions in chunks for float16 format
        let chunkSize = 4 // Process 4 positions at a time when possible
        let fullChunks = count / chunkSize
        
        for chunk in 0..<fullChunks {
            let baseIdx = chunk * chunkSize
            let sourceIdx = baseIdx * 6 // 2 bytes per component, 3 components
            
            // Process each component for the chunk
            for i in 0..<chunkSize {
                let pos = baseIdx + i
                for j in 0..<3 {
                    let idx = sourceIdx + i * 6 + j * 2
                    guard idx + 1 < packed.positions.count else { continue }
                    
                    let halfValue = UInt16(packed.positions[idx]) | (UInt16(packed.positions[idx + 1]) << 8)
                    result[pos * 3 + j] = float16ToFloat32(halfValue)
                }
            }
        }
        
        // Handle remaining positions
        for i in (fullChunks * chunkSize)..<count {
            let sourceIdx = i * 6
            for j in 0..<3 {
                let idx = sourceIdx + j * 2
                guard idx + 1 < packed.positions.count else { continue }
                
                let halfValue = UInt16(packed.positions[idx]) | (UInt16(packed.positions[idx + 1]) << 8)
                result[i * 3 + j] = float16ToFloat32(halfValue)
            }
        }
    } else {
        // Process fixed-point positions
        let scale = 1.0 / Float(1 << packed.fractionalBits)
        let chunkSize = 4 // Process 4 positions at a time when possible
        let fullChunks = count / chunkSize
        
        for chunk in 0..<fullChunks {
            let baseIdx = chunk * chunkSize
            let sourceIdx = baseIdx * 9 // 3 bytes per component, 3 components
            
            // Process each component for the chunk
            for i in 0..<chunkSize {
                let pos = baseIdx + i
                for j in 0..<3 {
                    let idx = sourceIdx + i * 9 + j * 3
                    guard idx + 2 < packed.positions.count else { continue }
                    
                    var fixed32: Int32 = Int32(packed.positions[idx])
                    fixed32 |= Int32(packed.positions[idx + 1]) << 8
                    fixed32 |= Int32(packed.positions[idx + 2]) << 16
                    if (fixed32 & 0x800000) != 0 {
                        fixed32 |= Int32(bitPattern: 0xFF000000) // Sign extension
                    }
                    result[pos * 3 + j] = Float(fixed32) * scale
                }
            }
        }
        
        // Handle remaining positions
        for i in (fullChunks * chunkSize)..<count {
            let sourceIdx = i * 9
            for j in 0..<3 {
                let idx = sourceIdx + j * 3
                guard idx + 2 < packed.positions.count else { continue }
                
                var fixed32: Int32 = Int32(packed.positions[idx])
                fixed32 |= Int32(packed.positions[idx + 1]) << 8
                fixed32 |= Int32(packed.positions[idx + 2]) << 16
                if (fixed32 & 0x800000) != 0 {
                    fixed32 |= Int32(bitPattern: 0xFF000000) // Sign extension
                }
                result[i * 3 + j] = Float(fixed32) * scale
            }
        }
    }
    
    return result
}

/// Process points in a cache-line friendly manner
private func processPointsByCacheLine(_ packed: PackedGaussians, result: inout GaussianCloud) {
    // Process points in cache-line sized chunks (64 bytes typically)
    let pointsPerCacheLine = 4 // 4 points at a time to match a typical 64-byte cache line
    let count = packed.numPoints
    
    for baseIdx in stride(from: 0, to: count, by: pointsPerCacheLine) {
        let endIdx = min(baseIdx + pointsPerCacheLine, count)
        
        // Process this cache-line sized chunk of points
        for pointIdx in baseIdx..<endIdx {
            let posIdx = pointIdx * 3
            
            if packed.usesFloat16 {
                let baseIn = pointIdx * 6  // 2 bytes per component
                for j in 0..<3 {
                    guard baseIn + j * 2 + 1 < packed.positions.count else { continue }
                    let halfValue = UInt16(packed.positions[baseIn + j * 2]) |
                                  (UInt16(packed.positions[baseIn + j * 2 + 1]) << 8)
                    result.positions[posIdx + j] = float16ToFloat32(halfValue)
                }
            } else {
                let baseIn = pointIdx * 9  // 3 bytes per component
                let scale = 1.0 / Float(1 << packed.fractionalBits)
                for j in 0..<3 {
                    guard baseIn + j * 3 + 2 < packed.positions.count else { continue }
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
