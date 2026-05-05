import Foundation
import Compression

// MARK: - Constants

private let ngspMagic: UInt32 = 0x5053474e  // "NGSP"
private let latestVersion: UInt32 = 4
private let minZstdVersion: UInt32 = 4       // v4+ uses ZSTD streams
private let flagAntialiased: UInt8 = 0x1
private let flagHasExtensions: UInt8 = 0x2
private let numV4Streams: UInt8 = 6

// MARK: - Header Structures

/// 16-byte header for SPZ versions 1–3 (inside a gzip envelope).
private struct LegacyHeader {
    static let size = 16

    var magic: UInt32
    var version: UInt32
    var numPoints: UInt32
    var shDegree: UInt8
    var fractionalBits: UInt8
    var flags: UInt8
    var reserved: UInt8

    init(data: Data) throws {
        guard data.count >= LegacyHeader.size else { throw SPZError.invalidHeader }
        magic          = data.withUnsafeBytes { $0.load(as: UInt32.self) }
        version        = data.withUnsafeBytes { $0.load(fromByteOffset: 4,  as: UInt32.self) }
        numPoints      = data.withUnsafeBytes { $0.load(fromByteOffset: 8,  as: UInt32.self) }
        shDegree       = data[12]
        fractionalBits = data[13]
        flags          = data[14]
        reserved       = data[15]
    }
}

/// 32-byte plaintext header for SPZ version 4 (ZSTD multi-stream).
private struct NgspHeaderV4 {
    static let size = 32

    var magic: UInt32          // offset  0
    var version: UInt32        // offset  4
    var numPoints: UInt32      // offset  8
    var shDegree: UInt8        // offset 12
    var fractionalBits: UInt8  // offset 13
    var flags: UInt8           // offset 14
    var numStreams: UInt8       // offset 15
    var tocByteOffset: UInt32  // offset 16
    // 12 reserved bytes at offsets 20–31

    init() {
        magic          = ngspMagic
        version        = latestVersion
        numPoints      = 0
        shDegree       = 0
        fractionalBits = 0
        flags          = 0
        numStreams      = numV4Streams
        tocByteOffset  = UInt32(NgspHeaderV4.size)  // no extensions
    }

    init(data: Data) throws {
        guard data.count >= NgspHeaderV4.size else { throw SPZError.invalidHeader }
        magic          = data.withUnsafeBytes { $0.load(as: UInt32.self) }
        version        = data.withUnsafeBytes { $0.load(fromByteOffset: 4,  as: UInt32.self) }
        numPoints      = data.withUnsafeBytes { $0.load(fromByteOffset: 8,  as: UInt32.self) }
        shDegree       = data[12]
        fractionalBits = data[13]
        flags          = data[14]
        numStreams      = data[15]
        tocByteOffset  = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt32.self) }
    }

    func serialize() -> Data {
        var out = Data(count: NgspHeaderV4.size)
        out.withUnsafeMutableBytes { rawBuf in
            let p = rawBuf.baseAddress!
            p.storeBytes(of: magic,         toByteOffset: 0,  as: UInt32.self)
            p.storeBytes(of: version,       toByteOffset: 4,  as: UInt32.self)
            p.storeBytes(of: numPoints,     toByteOffset: 8,  as: UInt32.self)
            p.storeBytes(of: shDegree,      toByteOffset: 12, as: UInt8.self)
            p.storeBytes(of: fractionalBits,toByteOffset: 13, as: UInt8.self)
            p.storeBytes(of: flags,         toByteOffset: 14, as: UInt8.self)
            p.storeBytes(of: numStreams,     toByteOffset: 15, as: UInt8.self)
            p.storeBytes(of: tocByteOffset, toByteOffset: 16, as: UInt32.self)
            // bytes 20-31 stay zero (reserved)
        }
        return out
    }
}

// MARK: - PackedGaussians serialise / deserialise (legacy v1–v3)

extension PackedGaussians {

    /// Serialises to the raw (uncompressed) binary layout used by v1–v3.
    func serializeLegacy() -> Data {
        var out = Data(capacity: LegacyHeader.size +
                       positions.count + alphas.count + colors.count +
                       scales.count + rotations.count + sh.count)
        // Write 16-byte legacy header
        func appendUInt32(_ v: UInt32) { withUnsafeBytes(of: v) { out.append(contentsOf: $0) } }
        appendUInt32(ngspMagic)
        appendUInt32(3)                                  // version 3 (smallest-three quaternions)
        appendUInt32(UInt32(numPoints))
        out.append(UInt8(shDegree))
        out.append(UInt8(fractionalBits))
        out.append(antialiased ? flagAntialiased : 0)
        out.append(0)                                    // reserved
        out.append(contentsOf: positions)
        out.append(contentsOf: alphas)
        out.append(contentsOf: colors)
        out.append(contentsOf: scales)
        out.append(contentsOf: rotations)
        out.append(contentsOf: sh)
        return out
    }

    /// Deserialises from the raw (post-gzip) binary layout of v1–v3.
    static func deserializeLegacy(_ data: Data) throws -> PackedGaussians {
        let header = try LegacyHeader(data: data)

        guard header.magic == ngspMagic             else { throw SPZError.invalidHeader }
        guard header.version >= 1 && header.version <= 3 else { throw SPZError.unsupportedVersion }
        guard header.shDegree <= 4                  else { throw SPZError.unsupportedSHDegree }

        let numPoints   = Int(header.numPoints)
        let shDim       = dimForDegree(Int(header.shDegree))
        let usesFloat16 = header.version == 1
        let usesST3     = header.version >= 3

        let posSize  = numPoints * 3 * (usesFloat16 ? 2 : 3)
        let alphasSz = numPoints
        let colorsSz = numPoints * 3
        let scalesSz = numPoints * 3
        let rotsSz   = numPoints * (usesST3 ? 4 : 3)
        let shSz     = numPoints * shDim * 3

        let expected = LegacyHeader.size + posSize + alphasSz + colorsSz + scalesSz + rotsSz + shSz
        guard data.count >= expected else { throw SPZError.invalidData }

        var result = PackedGaussians()
        result.numPoints                    = numPoints
        result.shDegree                     = Int(header.shDegree)
        result.fractionalBits               = Int(header.fractionalBits)
        result.antialiased                  = (header.flags & flagAntialiased) != 0
        result.usesQuaternionSmallestThree  = usesST3

        var offset = LegacyHeader.size
        func slice(_ size: Int) throws -> [UInt8] {
            guard offset + size <= data.count else { throw SPZError.invalidData }
            defer { offset += size }
            return Array(data[offset ..< offset + size])
        }

        result.positions  = try slice(posSize)
        result.alphas     = try slice(alphasSz)
        result.colors     = try slice(colorsSz)
        result.scales     = try slice(scalesSz)
        result.rotations  = try slice(rotsSz)
        result.sh         = try slice(shSz)
        return result
    }
}

// MARK: - Public API

public extension GaussianCloud {

    /// Saves the gaussian cloud to a compressed SPZ file (v4 ZSTD format).
    static func save(_ cloud: GaussianCloud,
                     to url: URL,
                     from coordinateSystem: CoordinateSystem = .unspecified) throws {
        let data = try save(cloud, from: coordinateSystem)
        try data.write(to: url)
    }

    /// Saves the gaussian cloud to compressed SPZ data (v4 ZSTD format).
    static func save(_ cloud: GaussianCloud,
                     from coordinateSystem: CoordinateSystem = .unspecified) throws -> Data {
        let packed = packGaussians(cloud, from: coordinateSystem)
        return try serializeV4(packed)
    }

    /// Loads a gaussian cloud from a compressed SPZ file.
    static func load(from url: URL,
                     to coordinateSystem: CoordinateSystem = .unspecified) throws -> GaussianCloud {
        let data = try Data(contentsOf: url)
        return try load(from: data, to: coordinateSystem)
    }

    /// Loads a gaussian cloud from compressed SPZ data (auto-detects v4 ZSTD or legacy gzip).
    static func load(from data: Data,
                     to coordinateSystem: CoordinateSystem = .unspecified) throws -> GaussianCloud {
        // v4: file begins with the NGSP magic in plaintext
        if data.count >= 4 {
            let firstWord = data.withUnsafeBytes { $0.load(as: UInt32.self) }
            if firstWord == ngspMagic {
                let packed = try deserializeV4(data)
                return unpackGaussians(packed, to: coordinateSystem)
            }
        }
        // v1–v3: gzip envelope containing the NGSP payload
        guard let decompressed = decompressGzipped(data) else {
            throw SPZError.decompressionError
        }
        let packed = try PackedGaussians.deserializeLegacy(decompressed)
        return unpackGaussians(packed, to: coordinateSystem)
    }
}

// MARK: - V4 Serialise / Deserialise

/// Builds a v4 SPZ byte stream: header + TOC + 6 ZSTD-compressed attribute streams.
private func serializeV4(_ packed: PackedGaussians) throws -> Data {
    // Compress the 6 attribute streams in parallel
    let streams: [Data] = [
        Data(packed.positions),
        Data(packed.alphas),
        Data(packed.colors),
        Data(packed.scales),
        Data(packed.rotations),
        Data(packed.sh),
    ]
    let n = streams.count
    var compressed = [Data?](repeating: nil, count: n)
    DispatchQueue.concurrentPerform(iterations: n) { i in
        compressed[i] = compressZstd(streams[i])
    }
    // Validate all streams compressed successfully
    for i in 0..<n {
        guard compressed[i] != nil else { throw SPZError.compressionError }
    }

    // Build header
    var header = NgspHeaderV4()
    header.numPoints      = UInt32(packed.numPoints)
    header.shDegree       = UInt8(packed.shDegree)
    header.fractionalBits = UInt8(packed.fractionalBits)
    header.flags          = packed.antialiased ? flagAntialiased : 0
    header.numStreams      = numV4Streams
    header.tocByteOffset  = UInt32(NgspHeaderV4.size)   // no extensions

    // Build TOC: numStreams × (compressedSize: UInt64, uncompressedSize: UInt64)
    let tocSize = Int(numV4Streams) * 16
    var toc = Data(count: tocSize)
    toc.withUnsafeMutableBytes { rawBuf in
        let p = rawBuf.baseAddress!
        for i in 0..<n {
            let cs = UInt64(compressed[i]!.count)
            let us = UInt64(streams[i].count)
            p.storeBytes(of: cs, toByteOffset: i * 16,     as: UInt64.self)
            p.storeBytes(of: us, toByteOffset: i * 16 + 8, as: UInt64.self)
        }
    }

    var out = header.serialize()
    out.append(toc)
    for c in compressed { out.append(c!) }
    return out
}

/// Parses a v4 SPZ byte stream into a `PackedGaussians`.
private func deserializeV4(_ data: Data) throws -> PackedGaussians {
    let header = try NgspHeaderV4(data: data)

    guard header.magic   == ngspMagic                         else { throw SPZError.invalidHeader }
    guard header.version >= minZstdVersion &&
          header.version <= latestVersion                      else { throw SPZError.unsupportedVersion }
    guard header.shDegree <= 4                                else { throw SPZError.unsupportedSHDegree }
    guard header.numPoints > 0                                else { throw SPZError.invalidData }

    let numPoints   = Int(header.numPoints)
    let shDim       = dimForDegree(Int(header.shDegree))
    let usesST3     = header.version >= 3
    let tocOffset   = Int(header.tocByteOffset)
    let numStreams   = Int(header.numStreams)

    guard numStreams == 6                                      else { throw SPZError.invalidData }
    let tocEnd = tocOffset + numStreams * 16
    guard data.count >= tocEnd                                 else { throw SPZError.invalidData }

    // Parse TOC
    struct StreamInfo { var compressedSize: Int; var uncompressedSize: Int }
    var infos = [StreamInfo](repeating: StreamInfo(compressedSize: 0, uncompressedSize: 0),
                             count: numStreams)
    data.withUnsafeBytes { rawBuf in
        let p = rawBuf.baseAddress!
        for i in 0..<numStreams {
            let base = tocOffset + i * 16
            let cs = p.load(fromByteOffset: base,     as: UInt64.self)
            let us = p.load(fromByteOffset: base + 8, as: UInt64.self)
            infos[i] = StreamInfo(compressedSize: Int(cs), uncompressedSize: Int(us))
        }
    }

    // Validate expected uncompressed sizes
    let expectedSizes: [Int] = [
        numPoints * 3 * 3,                        // positions (24-bit fixed-point)
        numPoints,                                 // alphas
        numPoints * 3,                             // colors
        numPoints * 3,                             // scales
        numPoints * (usesST3 ? 4 : 3),            // rotations
        numPoints * shDim * 3,                     // SH
    ]
    for i in 0..<numStreams {
        guard infos[i].uncompressedSize == expectedSizes[i] else { throw SPZError.invalidData }
    }

    // Decompress 6 streams in parallel
    var streamOffset = tocEnd
    var compressedSlices = [(Data, Int)]()  // (compressed bytes, uncompressed size)
    for i in 0..<numStreams {
        let end = streamOffset + infos[i].compressedSize
        guard end <= data.count else { throw SPZError.invalidData }
        compressedSlices.append((data[streamOffset ..< end], infos[i].uncompressedSize))
        streamOffset = end
    }

    var decompressed = [Data?](repeating: nil, count: numStreams)
    DispatchQueue.concurrentPerform(iterations: numStreams) { i in
        decompressed[i] = decompressZstd(compressedSlices[i].0,
                                         uncompressedSize: compressedSlices[i].1)
    }
    for i in 0..<numStreams {
        guard decompressed[i] != nil else { throw SPZError.decompressionError }
    }

    var result = PackedGaussians()
    result.numPoints                   = numPoints
    result.shDegree                    = Int(header.shDegree)
    result.fractionalBits              = Int(header.fractionalBits)
    result.antialiased                 = (header.flags & flagAntialiased) != 0
    result.usesQuaternionSmallestThree = usesST3
    result.positions  = [UInt8](decompressed[0]!)
    result.alphas     = [UInt8](decompressed[1]!)
    result.colors     = [UInt8](decompressed[2]!)
    result.scales     = [UInt8](decompressed[3]!)
    result.rotations  = [UInt8](decompressed[4]!)
    result.sh         = [UInt8](decompressed[5]!)
    return result
}

// MARK: - Unpack

public func unpackGaussians(_ packed: PackedGaussians,
                            to coordinateSystem: CoordinateSystem = .unspecified) -> GaussianCloud {
    let numPoints = packed.numPoints
    let shDim     = dimForDegree(packed.shDegree)

    guard checkSizes(packed, numPoints: numPoints, shDim: shDim) else {
        return GaussianCloud()
    }

    var result = GaussianCloud()
    result.numPoints  = numPoints
    result.shDegree   = packed.shDegree
    result.antialiased = packed.antialiased

    result.positions  = Array(repeating: 0, count: numPoints * 3)
    result.scales     = Array(repeating: 0, count: numPoints * 3)
    result.rotations  = Array(repeating: 0, count: numPoints * 4)
    result.alphas     = Array(repeating: 0, count: numPoints)
    result.colors     = Array(repeating: 0, count: numPoints * 3)
    result.sh         = Array(repeating: 0, count: numPoints * shDim * 3)

    let converter = coordinateSystem != .unspecified
        ? coordinateConverter(from: .rub, to: coordinateSystem)
        : nil

    let scale      = 1.0 / Float(1 << packed.fractionalBits)
    let usesST3    = packed.usesQuaternionSmallestThree
    let shStride   = shDim * 3

    for i in 0..<numPoints {
        let i3 = i * 3
        let i4 = i * 4

        // Position
        let flipP = converter?.flipP ?? Vec3f(1, 1, 1)
        if packed.usesFloat16 {
            let posBase = i * 6
            for j in 0..<3 {
                let h = UInt16(packed.positions[posBase + j * 2])
                      | UInt16(packed.positions[posBase + j * 2 + 1]) << 8
                result.positions[i3 + j] = float16ToFloat32(h) * flipP[j]
            }
        } else {
            let posBase = i * 9
            for j in 0..<3 {
                var v: Int32 = Int32(packed.positions[posBase + j * 3])
                v |= Int32(packed.positions[posBase + j * 3 + 1]) << 8
                v |= Int32(packed.positions[posBase + j * 3 + 2]) << 16
                if (v & 0x800000) != 0 { v |= Int32(bitPattern: 0xFF000000) }
                result.positions[i3 + j] = Float(v) * scale * flipP[j]
            }
        }

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
        var q = Quat4f(0, 0, 0, 1)
        if usesST3 {
            let rot4 = Array(packed.rotations[i4 ..< i4 + 4])
            unpackQuaternionSmallestThree(&q, rot4, converter ?? CoordinateConverter())
        } else {
            let rot3 = Array(packed.rotations[i3 ..< i3 + 3])
            unpackQuaternionFirstThree(&q, rot3, converter ?? CoordinateConverter())
        }
        result.rotations[i4 + 0] = q.x
        result.rotations[i4 + 1] = q.y
        result.rotations[i4 + 2] = q.z
        result.rotations[i4 + 3] = q.w

        // SH
        let shBase = i * shStride
        let flipSh = converter?.flipSh ?? Array(repeating: 1.0, count: 24)
        for j in 0..<shDim {
            let flip = j < flipSh.count ? flipSh[j] : 1.0
            let base = shBase + j * 3
            result.sh[base + 0] = flip * unquantizeSH(packed.sh[base + 0])
            result.sh[base + 1] = flip * unquantizeSH(packed.sh[base + 1])
            result.sh[base + 2] = flip * unquantizeSH(packed.sh[base + 2])
        }
    }

    return result
}

// MARK: - Pack

/// Packs quaternion using smallest-three encoding into 4 bytes.
private func packQuaternionSmallestThree(_ q: Quat4f, _ rotations: inout [UInt8], _ offset: Int) {
    let normalized = q.normalized
    let absVals = [abs(normalized.x), abs(normalized.y), abs(normalized.z), abs(normalized.w)]
    let largestIdx = absVals.enumerated().max(by: { $0.element < $1.element })!.offset

    var qn = normalized
    if qn[largestIdx] < 0 { qn = -qn }

    var smallestThree: [Float] = []
    for i in 0..<4 where i != largestIdx { smallestThree.append(qn[i]) }

    var comp: UInt32 = UInt32(largestIdx)
    for i in 0..<3 {
        let val     = smallestThree[i]
        let negbit: UInt32 = val < 0 ? 1 : 0
        let mag     = UInt32(Float((1 << 9) - 1) * (abs(val) / sqrt1_2) + 0.5)
        comp = (comp << 10) | (negbit << 9) | mag
    }

    rotations[offset + 0] = UInt8( comp        & 0xFF)
    rotations[offset + 1] = UInt8((comp >>  8) & 0xFF)
    rotations[offset + 2] = UInt8((comp >> 16) & 0xFF)
    rotations[offset + 3] = UInt8((comp >> 24) & 0xFF)
}

private func packGaussians(_ cloud: GaussianCloud,
                           from coordinateSystem: CoordinateSystem = .unspecified) -> PackedGaussians {
    guard checkSizes(cloud) else { return PackedGaussians() }

    let numPoints = cloud.numPoints
    let shDim     = dimForDegree(cloud.shDegree)

    var src = cloud
    if coordinateSystem != .unspecified {
        src.convertCoordinates(from: coordinateSystem, to: .rub)
    }

    var packed = PackedGaussians()
    packed.numPoints                   = numPoints
    packed.shDegree                    = cloud.shDegree
    packed.fractionalBits              = 12
    packed.antialiased                 = cloud.antialiased
    packed.usesQuaternionSmallestThree = true

    packed.positions  = Array(repeating: 0, count: numPoints * 9)
    packed.scales     = Array(repeating: 0, count: numPoints * 3)
    packed.rotations  = Array(repeating: 0, count: numPoints * 4)
    packed.alphas     = Array(repeating: 0, count: numPoints)
    packed.colors     = Array(repeating: 0, count: numPoints * 3)
    packed.sh         = Array(repeating: 0, count: numPoints * shDim * 3)

    let fscale = Float(1 << packed.fractionalBits)
    for i in 0..<numPoints * 3 {
        let v = src.positions[i]
        let s = v.isFinite ? v : 0
        let f = Int32(round(s * fscale))
        packed.positions[i * 3 + 0] = UInt8( f        & 0xFF)
        packed.positions[i * 3 + 1] = UInt8((f >>  8) & 0xFF)
        packed.positions[i * 3 + 2] = UInt8((f >> 16) & 0xFF)
    }

    for i in 0..<numPoints * 3 {
        let v = src.scales[i]
        packed.scales[i] = toUInt8(((v.isFinite ? v : 0) + 10.0) * 16.0)
    }

    for i in 0..<numPoints {
        let q = Quat4f(
            src.rotations[i*4+0].isFinite ? src.rotations[i*4+0] : 0,
            src.rotations[i*4+1].isFinite ? src.rotations[i*4+1] : 0,
            src.rotations[i*4+2].isFinite ? src.rotations[i*4+2] : 0,
            src.rotations[i*4+3].isFinite ? src.rotations[i*4+3] : 1
        ).normalized
        packQuaternionSmallestThree(q, &packed.rotations, i * 4)
    }

    for i in 0..<numPoints {
        let v = src.alphas[i]
        packed.alphas[i] = toUInt8(sigmoid(v.isFinite ? v : 0) * 255.0)
    }

    for i in 0..<numPoints * 3 {
        let v = src.colors[i]
        packed.colors[i] = toUInt8((v.isFinite ? v : 0) * (colorScale * 255.0) + (0.5 * 255.0))
    }

    if cloud.shDegree > 0 {
        let sh1Bits    = 5
        let shRestBits = 4
        let shPerPoint = shDim * 3

        for i in stride(from: 0, to: numPoints * shPerPoint, by: shPerPoint) {
            // First 9 entries = degree-1 coefficients (3 coeff × 3 channels)
            for j in 0..<min(9, shPerPoint) {
                let v = src.sh[i + j]
                packed.sh[i + j] = quantizeSH(v.isFinite ? v : 0, bucketSize: 1 << (8 - sh1Bits))
            }
            for j in 9..<shPerPoint {
                let v = src.sh[i + j]
                packed.sh[i + j] = quantizeSH(v.isFinite ? v : 0, bucketSize: 1 << (8 - shRestBits))
            }
        }
    }

    return packed
}

// MARK: - Float16 Support

private func float16ToFloat32(_ half: UInt16) -> Float {
    let sign     = (half >> 15) & 0x1
    let exponent = (half >> 10) & 0x1f
    let mantissa =  half        & 0x3ff
    let signMul: Float = sign == 1 ? -1.0 : 1.0
    if exponent == 0  { return signMul * Float(pow(2.0, -14.0)) * Float(mantissa) / 1024.0 }
    if exponent == 31 { return mantissa != 0 ? Float.nan : signMul * Float.infinity }
    return signMul * Float(pow(2.0, Double(exponent) - 15.0)) * (1.0 + Float(mantissa) / 1024.0)
}

// MARK: - Size Checks

private func checkSizes(_ cloud: GaussianCloud) -> Bool {
    guard cloud.shDegree >= 0 && cloud.shDegree <= 4                          else { return false }
    guard cloud.positions.count  == cloud.numPoints * 3                       else { return false }
    guard cloud.scales.count     == cloud.numPoints * 3                       else { return false }
    guard cloud.rotations.count  == cloud.numPoints * 4                       else { return false }
    guard cloud.alphas.count     == cloud.numPoints                           else { return false }
    guard cloud.colors.count     == cloud.numPoints * 3                       else { return false }
    guard cloud.sh.count         == cloud.numPoints * dimForDegree(cloud.shDegree) * 3 else { return false }
    return true
}

private func checkSizes(_ packed: PackedGaussians, numPoints: Int, shDim: Int) -> Bool {
    let usesST3  = packed.usesQuaternionSmallestThree
    let posBytes = packed.usesFloat16 ? 2 : 3  // bytes per position component
    guard packed.positions.count == numPoints * 3 * posBytes           else { return false }
    guard packed.scales.count    == numPoints * 3                      else { return false }
    guard packed.rotations.count == numPoints * (usesST3 ? 4 : 3)     else { return false }
    guard packed.alphas.count    == numPoints                          else { return false }
    guard packed.colors.count    == numPoints * 3                      else { return false }
    guard packed.sh.count        == numPoints * shDim * 3              else { return false }
    return true
}

// MARK: - Quaternion helpers (used by unpackGaussians)

private func unpackQuaternionFirstThree(_ result: inout Quat4f,
                                        _ rotation: [UInt8],
                                        _ c: CoordinateConverter) {
    guard rotation.count >= 3 else { return }
    let xyz = Vec3f(Float(rotation[0]), Float(rotation[1]), Float(rotation[2])) / 127.5
             - Vec3f(1, 1, 1)
    let flipped = Vec3f(xyz.x * c.flipQ.x, xyz.y * c.flipQ.y, xyz.z * c.flipQ.z)
    result.x = flipped.x
    result.y = flipped.y
    result.z = flipped.z
    result.w = sqrt(max(0, 1.0 - flipped.squaredNorm))
}

private func unpackQuaternionSmallestThree(_ result: inout Quat4f,
                                           _ rotation: [UInt8],
                                           _ c: CoordinateConverter) {
    guard rotation.count >= 4 else { return }

    // Pack layout (little-endian 32-bit word, MSB first):
    //  bits 31-30: largestIdx (2 bits)
    //  bits 29-20: component 0 (sign bit + 9-bit magnitude)
    //  bits 19-10: component 1
    //  bits  9- 0: component 2
    let word = UInt32(rotation[0]) | (UInt32(rotation[1]) << 8)
             | (UInt32(rotation[2]) << 16) | (UInt32(rotation[3]) << 24)

    let largestIdx = Int(word >> 30)

    var components = [Float](repeating: 0, count: 4)
    var compIdx = 0
    for i in 0..<4 where i != largestIdx {
        let shift = 20 - compIdx * 10
        let bits  = (word >> shift) & 0x3FF
        let negbit = (bits >> 9) & 0x1
        let mag    = Float(bits & 0x1FF)
        let value  = (negbit != 0 ? -1.0 : 1.0) * mag / Float((1 << 9) - 1) * sqrt1_2
        components[i] = value * (i < 3 ? c.flipQ[i] : 1.0)  // w (i=3) is never flipped
        compIdx += 1
    }
    let sumSq = components[0]*components[0] + components[1]*components[1]
              + components[2]*components[2] + components[3]*components[3]
    components[largestIdx] = sqrt(max(0, 1.0 - sumSq))

    result.x = components[0]
    result.y = components[1]
    result.z = components[2]
    result.w = components[3]
}
