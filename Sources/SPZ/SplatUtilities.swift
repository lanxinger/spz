import Foundation
import Compression
import zlib

// MARK: - Constants

/// Scale factor for DC color components
public let colorScale: Float = 0.15

// MARK: - Math Utilities

/// Sigmoid activation function
public func sigmoid(_ x: Float) -> Float {
    1.0 / (1.0 + exp(-x))
}

/// Inverse sigmoid function
public func invSigmoid(_ x: Float) -> Float {
    log(x / (1.0 - x))
}

/// Converts a float to UInt8, clamping to [0, 255]
public func toUInt8(_ x: Float) -> UInt8 {
    UInt8(max(0, min(255, round(x))))
}

/// Quantizes a value to 8 bits with specified bucket size
public func quantizeSH(_ x: Float, bucketSize: Int) -> UInt8 {
    let q = Int(round(x * 128.0) + 128.0)
    let quantized = ((q + bucketSize / 2) / bucketSize * bucketSize)
    return UInt8(max(0, min(255, quantized)))
}

/// Converts a quantized SH value back to float
public func unquantizeSH(_ x: UInt8) -> Float {
    (Float(x) - 128.0) / 128.0
}

/// Returns the dimension for a given spherical harmonics degree
public func dimForDegree(_ degree: Int) -> Int {
    switch degree {
    case 0: return 0
    case 1: return 3
    case 2: return 8
    case 3: return 15
    default:
        print("[SPZ: ERROR] Unsupported SH degree: \(degree)")
        return 0
    }
}

/// Returns the degree for a given spherical harmonics dimension
public func degreeForDim(_ dim: Int) -> Int {
    if dim < 3 { return 0 }
    if dim < 8 { return 1 }
    if dim < 15 { return 2 }
    return 3
}

// MARK: - Compression Utilities

/// Use larger buffer sizes for compression
private let compressionBufferSize = 64 * 1024  // 64KB buffer

/// Compresses data using gzip compression
public func compressGzipped(_ data: Data) -> Data? {
    let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: compressionBufferSize)
    defer { destinationBuffer.deallocate() }
    
    let algorithm = COMPRESSION_ZLIB
    
    return data.withUnsafeBytes { sourceBuffer in
        guard let sourcePtr = sourceBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
            return nil
        }
        
        let sourceSize = data.count
        let result = compression_encode_buffer(
            destinationBuffer,
            sourceSize,
            sourcePtr,
            sourceSize,
            nil,
            algorithm
        )
        
        guard result > 0 else { return nil }
        return Data(bytes: destinationBuffer, count: result)
    }
}

/// Decompresses gzipped data
public func decompressGzipped(_ data: Data) -> Data? {
    print("[SPZ] Attempting to decompress \(data.count) bytes")
    
    // Check for gzip magic number (1f 8b)
    if data.count >= 2 {
        let magic1 = data[0]
        let magic2 = data[1]
        print("[SPZ] File magic: \(String(format: "%02x %02x", magic1, magic2))")
    }
    
    // Start with a reasonable buffer size (16MB)
    var destinationBuffer = [UInt8](repeating: 0, count: 16 * 1024 * 1024)
    var stream = z_stream()
    
    // Initialize for gzip decoding (16 + MAX_WBITS for gzip)
    let initResult = withUnsafeMutablePointer(to: &stream) { streamPtr in
        inflateInit2_(streamPtr, 16 + MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
    }
    
    guard initResult == Z_OK else {
        print("[SPZ] Error: Failed to initialize decompression: \(initResult)")
        return nil
    }
    
    defer {
        withUnsafeMutablePointer(to: &stream) { streamPtr in
            _ = inflateEnd(streamPtr)
        }
    }
    
    // Set up source buffer
    return data.withUnsafeBytes { sourceBuffer -> Data? in
        guard let sourcePtr = sourceBuffer.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
            print("[SPZ] Error: Failed to get source buffer address")
            return nil
        }
        
        stream.next_in = UnsafeMutablePointer(mutating: sourcePtr)
        stream.avail_in = UInt32(data.count)
        
        var decompressed = Data()
        
        repeat {
            destinationBuffer.withUnsafeMutableBufferPointer { buffer in
                stream.next_out = buffer.baseAddress
                stream.avail_out = UInt32(buffer.count)
            }
            
            let inflateResult = withUnsafeMutablePointer(to: &stream) { streamPtr in
                inflate(streamPtr, Z_NO_FLUSH)
            }
            
            if inflateResult != Z_OK && inflateResult != Z_STREAM_END {
                print("[SPZ] Error: Failed to decompress: \(inflateResult)")
                return nil
            }
            
            let bytesDecompressed = destinationBuffer.count - Int(stream.avail_out)
            decompressed.append(destinationBuffer, count: bytesDecompressed)
            
            if inflateResult == Z_STREAM_END {
                break
            }
            
        } while stream.avail_out == 0
        
        print("[SPZ] Successfully decompressed to \(decompressed.count) bytes")
        
        // Check header magic (NGSP = 0x5053474e)
        if decompressed.count >= 4 {
            let magic = decompressed.withUnsafeBytes { $0.load(as: UInt32.self) }
            print("[SPZ] Decompressed header magic: \(String(format: "0x%08x", magic))")
        }
        
        return decompressed
    }
}

// Memory pool for temporary buffers
private final class MemoryPool {
    private var buffers: [[UInt8]] = []
    private let lock = NSLock()
    
    func acquire(size: Int) -> [UInt8] {
        lock.lock()
        defer { lock.unlock() }
        
        if let index = buffers.firstIndex(where: { $0.count >= size }) {
            let buffer = buffers[index]
            buffers.remove(at: index)
            return buffer
        }
        
        return [UInt8](repeating: 0, count: size)
    }
    
    func release(_ buffer: [UInt8]) {
        lock.lock()
        buffers.append(buffer)
        lock.unlock()
    }
}

private let sharedMemoryPool = MemoryPool()

/// Compresses data using gzip compression in parallel
public func compressGzippedParallel(_ data: Data, chunkSize: Int = 1024 * 1024) -> Data? {
    let chunks = stride(from: 0, to: data.count, by: chunkSize).map {
        data[$0..<min($0 + chunkSize, data.count)]
    }
    
    var compressedChunks: [Data?] = Array(repeating: nil, count: chunks.count)
    
    DispatchQueue.concurrentPerform(iterations: chunks.count) { i in
        compressedChunks[i] = compressGzipped(chunks[i])
    }
    
    guard !compressedChunks.contains(where: { $0 == nil }) else { return nil }
    
    var result = Data()
    compressedChunks.forEach { chunk in
        if let chunk = chunk {
            result.append(chunk)
        }
    }
    return result
}