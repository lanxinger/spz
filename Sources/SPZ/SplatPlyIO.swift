import Foundation

extension GaussianCloud {
    /// Loads a gaussian cloud from a PLY file
    public static func loadFromPly(url: URL, to coordinateSystem: CoordinateSystem = .unspecified) throws -> GaussianCloud {
        print("[SPZ] Loading: \(url.path)")
        
        // First try to read the header as text
        let fileHandle = try FileHandle(forReadingFrom: url)
        defer { try? fileHandle.close() }
        
        // Read first few bytes to check format
        let headerBuffer = readHeaderBuffer(fileHandle: fileHandle)
        guard let headerStart = String(data: headerBuffer, encoding: .ascii) else {
            throw SPZError.invalidFormat("Not a valid PLY file")
        }
        
        // Split into lines, keeping only valid ASCII lines
        let headerLines = headerStart.split(separator: "\n")
            .map(String.init)
            .filter { $0.allSatisfy { $0.isASCII } }
        
        guard !headerLines.isEmpty && headerLines[0] == "ply" else {
            throw SPZError.invalidFormat("Not a .ply file")
        }
        
        // Check format
        let formatLine = headerLines.first { $0.starts(with: "format ") }
        guard let formatLine = formatLine else {
            throw SPZError.invalidFormat("Missing format specification")
        }
        
        let format = formatLine.dropFirst("format ".count)
        guard format.starts(with: "binary_little_endian 1.0") else {
            throw SPZError.invalidFormat("Unsupported PLY format: \(format)")
        }
        
        // Parse header
        var numPoints = 0
        var fields: [String: Int] = [:]  // name -> index
        var headerEndOffset: UInt64 = 0
        
        // Read the file line by line until we find end_header
        try fileHandle.seek(toOffset: 0)
        var headerEnded = false
        var currentOffset: UInt64 = 0
        
        while !headerEnded {
            guard let line = fileHandle.readLine() else {
                throw SPZError.invalidFormat("Unexpected end of file while reading header")
            }
            currentOffset += UInt64(line.count + 1)  // +1 for newline
            
            if line.starts(with: "element vertex ") {
                numPoints = Int(line.dropFirst("element vertex ".count)) ?? 0
                guard numPoints > 0 && numPoints <= 10 * 1024 * 1024 else {
                    throw SPZError.invalidFormat("Invalid vertex count: \(numPoints)")
                }
            } else if line.starts(with: "property float ") {
                let name = String(line.dropFirst("property float ".count))
                fields[name] = fields.count
            } else if line == "end_header" {
                headerEndOffset = currentOffset
                headerEnded = true
            }
        }
        
        print("[SPZ] Loading \(numPoints) points")
        
        // Helper to get field index
        func fieldIndex(_ name: String) throws -> Int {
            guard let index = fields[name] else {
                throw SPZError.invalidFormat("Missing field: \(name)")
            }
            return index
        }
        
        // Get field indices
        let positionIdx = try [
            fieldIndex("x"),
            fieldIndex("y"),
            fieldIndex("z")
        ]
        
        let scaleIdx = try [
            fieldIndex("scale_0"),
            fieldIndex("scale_1"),
            fieldIndex("scale_2")
        ]
        
        let rotIdx = try [
            fieldIndex("rot_1"),
            fieldIndex("rot_2"),
            fieldIndex("rot_3"),
            fieldIndex("rot_0")
        ]
        
        let alphaIdx = try [fieldIndex("opacity")]
        
        let colorIdx = try [
            fieldIndex("f_dc_0"),
            fieldIndex("f_dc_1"),
            fieldIndex("f_dc_2")
        ]
        
        // Get optional SH indices
        var shIdx: [Int] = []
        for i in 0..<45 {
            if let idx = fields["f_rest_\(i)"] {
                shIdx.append(idx)
            } else {
                break
            }
        }
        let shDim = shIdx.count / 3
        
        // Read binary data
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw SPZError.readError
        }
        defer { try? fileHandle.close() }
        
        // Skip header
        try fileHandle.seek(toOffset: headerEndOffset)
        
        // Read float values
        let floatSize = MemoryLayout<Float>.size
        let valuesPerPoint = fields.count
        let dataSize = numPoints * valuesPerPoint * floatSize
        
        let binaryData: Data
        if #available(iOS 13.4, macOS 10.15.4, *) {
            guard let readData = try? fileHandle.read(upToCount: dataSize),
                  readData.count == dataSize else {
                throw SPZError.readError
            }
            binaryData = readData
        } else {
            // Fallback for older versions
            let readData = fileHandle.readData(ofLength: dataSize)
            guard readData.count == dataSize else {
                throw SPZError.readError
            }
            binaryData = readData
        }
        
        // Create a local copy of the values array to avoid overlapping access
        let valuesData = binaryData.withUnsafeBytes { ptr -> [Float] in
            guard let floatPtr = ptr.bindMemory(to: Float.self).baseAddress else {
                return Array(repeating: 0, count: numPoints * valuesPerPoint)
            }
            // Create array first with explicit type
            let values: [Float] = Array(unsafeUninitializedCapacity: numPoints * valuesPerPoint) { buffer, initializedCount in
                buffer.baseAddress!.initialize(from: floatPtr, count: numPoints * valuesPerPoint)
                initializedCount = numPoints * valuesPerPoint
            }
            return values
        }
        
        // Create gaussian cloud
        var result = GaussianCloud()
        result.numPoints = numPoints
        result.shDegree = degreeForDim(shDim)
        
        // Pre-allocate arrays
        result.positions = Array(repeating: 0, count: numPoints * 3)
        result.scales = Array(repeating: 0, count: numPoints * 3)
        result.rotations = Array(repeating: 0, count: numPoints * 4)
        result.alphas = Array(repeating: 0, count: numPoints)
        result.colors = Array(repeating: 0, count: numPoints * 3)
        result.sh = Array(repeating: 0, count: numPoints * shDim * 3)
        
        // Progress tracking
        let progressInterval = max(1, numPoints / 100)  // Update every 1%
        var lastProgress = 0
        
        // Copy data to result
        for i in 0..<numPoints {
            let offset = i * valuesPerPoint
            
            // Update progress
            let progress = (i * 100) / numPoints
            if progress != lastProgress && i % progressInterval == 0 {
                print("\r[SPZ] Loading progress: \(progress)%", terminator: "")
                fflush(stdout)
                lastProgress = progress
            }
            
            // Copy positions
            for j in 0..<3 {
                result.positions[i * 3 + j] = valuesData[offset + positionIdx[j]]
            }
            
            // Copy scales
            for j in 0..<3 {
                result.scales[i * 3 + j] = valuesData[offset + scaleIdx[j]]
            }
            
            // Copy rotations
            for j in 0..<4 {
                result.rotations[i * 4 + j] = valuesData[offset + rotIdx[j]]
            }
            
            // Copy alpha
            result.alphas[i] = valuesData[offset + alphaIdx[0]]
            
            // Copy colors
            for j in 0..<3 {
                result.colors[i * 3 + j] = valuesData[offset + colorIdx[j]]
            }
            
            // Copy spherical harmonics
            let shOffset = i * shDim * 3
            for j in 0..<shDim {
                result.sh[shOffset + j * 3 + 0] = valuesData[offset + shIdx[j]]
                result.sh[shOffset + j * 3 + 1] = valuesData[offset + shIdx[j + shDim]]
                result.sh[shOffset + j * 3 + 2] = valuesData[offset + shIdx[j + 2 * shDim]]
            }
        }
        
        // Convert coordinates if needed (PLY is in RDF coordinate system)
        if coordinateSystem != .unspecified {
            result.convertCoordinates(from: .rdf, to: coordinateSystem)
        }
        
        print("\r[SPZ] Loading complete: \(numPoints) points")  // Clear progress line with final message
        return result
    }
    
    /// Saves the gaussian cloud to a PLY file
    public func saveToPly(url: URL, from coordinateSystem: CoordinateSystem = .unspecified) throws {
        let N = numPoints
        let shDim = sh.count / (N * 3)
        let D = 17 + shDim * 3
        
        print("[SPZ] Saving \(N) points to PLY file")
        
        // Validate sizes
        guard positions.count == N * 3,
              scales.count == N * 3,
              rotations.count == N * 4,
              alphas.count == N,
              colors.count == N * 3 else {
            throw SPZError.invalidData
        }
        
        // Create a copy for coordinate conversion if needed
        var cloudToConvert = self
        if coordinateSystem != .unspecified {
            cloudToConvert.convertCoordinates(from: coordinateSystem, to: .rdf)
        }
        
        var values = Array(repeating: Float(0), count: N * D)
        var outIdx = 0
        
        // Progress tracking
        let progressInterval = max(1, N / 100)  // Update every 1%
        var lastProgress = 0
        
        for i in 0..<N {
            // Update progress
            let progress = (i * 100) / N
            if progress != lastProgress && i % progressInterval == 0 {
                print("[SPZ] Saving progress: \(progress)%")
                lastProgress = progress
            }
            
            let i3 = i * 3
            let i4 = i * 4
            
            // Position (x, y, z)
            values[outIdx] = cloudToConvert.positions[i3]; outIdx += 1
            values[outIdx] = cloudToConvert.positions[i3 + 1]; outIdx += 1
            values[outIdx] = cloudToConvert.positions[i3 + 2]; outIdx += 1
            
            // Normals (nx, ny, nz) - always zero
            outIdx += 3
            
            // Color (r, g, b)
            values[outIdx] = cloudToConvert.colors[i3]; outIdx += 1
            values[outIdx] = cloudToConvert.colors[i3 + 1]; outIdx += 1
            values[outIdx] = cloudToConvert.colors[i3 + 2]; outIdx += 1
            
            // Spherical harmonics
            for j in 0..<shDim {
                values[outIdx] = cloudToConvert.sh[(i * shDim + j) * 3]; outIdx += 1
            }
            for j in 0..<shDim {
                values[outIdx] = cloudToConvert.sh[(i * shDim + j) * 3 + 1]; outIdx += 1
            }
            for j in 0..<shDim {
                values[outIdx] = cloudToConvert.sh[(i * shDim + j) * 3 + 2]; outIdx += 1
            }
            
            // Alpha
            values[outIdx] = cloudToConvert.alphas[i]; outIdx += 1
            
            // Scale
            values[outIdx] = cloudToConvert.scales[i3]; outIdx += 1
            values[outIdx] = cloudToConvert.scales[i3 + 1]; outIdx += 1
            values[outIdx] = cloudToConvert.scales[i3 + 2]; outIdx += 1
            
            // Rotation
            values[outIdx] = cloudToConvert.rotations[i4 + 3]; outIdx += 1  // w
            values[outIdx] = cloudToConvert.rotations[i4]; outIdx += 1      // x
            values[outIdx] = cloudToConvert.rotations[i4 + 1]; outIdx += 1  // y
            values[outIdx] = cloudToConvert.rotations[i4 + 2]; outIdx += 1  // z
        }
        
        // Write header
        var header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "element vertex \(N)\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        header += "property float nx\n"
        header += "property float ny\n"
        header += "property float nz\n"
        header += "property float f_dc_0\n"
        header += "property float f_dc_1\n"
        header += "property float f_dc_2\n"
        
        for i in 0..<(shDim * 3) {
            header += "property float f_rest_\(i)\n"
        }
        
        header += "property float opacity\n"
        header += "property float scale_0\n"
        header += "property float scale_1\n"
        header += "property float scale_2\n"
        header += "property float rot_0\n"
        header += "property float rot_1\n"
        header += "property float rot_2\n"
        header += "property float rot_3\n"
        header += "end_header\n"
        
        // Write file
        guard let headerData = header.data(using: .utf8) else {
            throw SPZError.writeError
        }
        
        var fileData = Data()
        fileData.append(headerData)
        values.withUnsafeBytes { ptr in
            fileData.append(ptr.baseAddress!.assumingMemoryBound(to: UInt8.self), count: ptr.count)
        }
        
        try fileData.write(to: url)
        
        print("[SPZ] Save complete: \(N) points written to \(url.path)")
    }
}

// MARK: - FileHandle Extension

extension FileHandle {
    func readLine() -> String? {
        var line = Data()
        let bufferSize = 1
        
        while true {
            let data: Data
            if #available(iOS 13.4, macOS 10.15.4, *) {
                guard let read = try? self.read(upToCount: bufferSize) else { break }
                data = read
            } else {
                data = self.readData(ofLength: bufferSize)
            }
            
            if data.isEmpty { break }
            
            if let byte = data.first {
                if byte == UInt8(ascii: "\n") {
                    break
                }
                line.append(byte)
            }
        }
        
        return String(data: line, encoding: .ascii)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// Add this function at the top of the file, before loadFromPly
private func readHeaderBuffer(fileHandle: FileHandle) -> Data {
    if #available(iOS 13.4, macOS 10.15.4, *) {
        return (try? fileHandle.read(upToCount: 1024)) ?? Data()
    } else {
        return fileHandle.readData(ofLength: 1024)
    }
} 