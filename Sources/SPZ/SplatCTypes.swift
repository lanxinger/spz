import Foundation

/// C-compatible buffer type for float arrays
public struct FloatBuffer {
    public var count: Int
    public var data: UnsafeMutablePointer<Float>?
    
    public init() {
        count = 0
        data = nil
    }
    
    public init(array: [Float]) {
        count = array.count
        if !array.isEmpty {
            data = UnsafeMutablePointer<Float>.allocate(capacity: count)
            data!.initialize(from: array, count: count)
        } else {
            data = nil
        }
    }
    
    public mutating func deallocate() {
        data?.deallocate()
        data = nil
        count = 0
    }
    
    public func toArray() -> [Float] {
        guard let data = data, count > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: data, count: count))
    }
}

/// C-compatible data structure for gaussian cloud
public struct GaussianCloudData {
    public var numPoints: Int32 = 0
    public var shDegree: Int32 = 0
    public var antialiased: Bool = false
    public var positions: FloatBuffer = FloatBuffer()
    public var scales: FloatBuffer = FloatBuffer()
    public var rotations: FloatBuffer = FloatBuffer()
    public var alphas: FloatBuffer = FloatBuffer()
    public var colors: FloatBuffer = FloatBuffer()
    public var sh: FloatBuffer = FloatBuffer()
    
    public init() {}
    
    public init(from cloud: GaussianCloud) {
        numPoints = Int32(cloud.numPoints)
        shDegree = Int32(cloud.shDegree)
        antialiased = cloud.antialiased
        positions = FloatBuffer(array: cloud.positions)
        scales = FloatBuffer(array: cloud.scales)
        rotations = FloatBuffer(array: cloud.rotations)
        alphas = FloatBuffer(array: cloud.alphas)
        colors = FloatBuffer(array: cloud.colors)
        sh = FloatBuffer(array: cloud.sh)
    }
    
    public func toGaussianCloud() -> GaussianCloud {
        var cloud = GaussianCloud()
        cloud.numPoints = Int(numPoints)
        cloud.shDegree = Int(shDegree)
        cloud.antialiased = antialiased
        cloud.positions = positions.toArray()
        cloud.scales = scales.toArray()
        cloud.rotations = rotations.toArray()
        cloud.alphas = alphas.toArray()
        cloud.colors = colors.toArray()
        cloud.sh = sh.toArray()
        return cloud
    }
    
    public mutating func deallocate() {
        positions.deallocate()
        scales.deallocate()
        rotations.deallocate()
        alphas.deallocate()
        colors.deallocate()
        sh.deallocate()
    }
}

/// C-compatible pack options
public struct PackOptionsC {
    public var from: Int32 = 0  // CoordinateSystem.unspecified.rawValue
    
    public init() {}
    
    public init(from coordinateSystem: CoordinateSystem) {
        from = Int32(coordinateSystem.rawValue)
    }
    
    public var coordinateSystem: CoordinateSystem {
        CoordinateSystem(rawValue: Int(from)) ?? .unspecified
    }
}

/// C-compatible unpack options
public struct UnpackOptionsC {
    public var to: Int32 = 0  // CoordinateSystem.unspecified.rawValue
    
    public init() {}
    
    public init(to coordinateSystem: CoordinateSystem) {
        to = Int32(coordinateSystem.rawValue)
    }
    
    public var coordinateSystem: CoordinateSystem {
        CoordinateSystem(rawValue: Int(to)) ?? .unspecified
    }
}

// MARK: - C-compatible API Functions

/// C-compatible save function that returns data as bytes
public func saveSPZC(gaussians: GaussianCloudData, options: PackOptionsC) -> (data: UnsafeMutablePointer<UInt8>?, size: Int) {
    let cloud = gaussians.toGaussianCloud()
    
    do {
        let data = try GaussianCloud.save(cloud, from: options.coordinateSystem)
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
        _ = data.copyBytes(to: UnsafeMutableBufferPointer(start: buffer, count: data.count))
        return (buffer, data.count)
    } catch {
        return (nil, 0)
    }
}

/// C-compatible load function that takes bytes
public func loadSPZC(data: UnsafePointer<UInt8>, size: Int, options: UnpackOptionsC) -> GaussianCloudData {
    let nsData = Data(bytes: data, count: size)
    
    do {
        let cloud = try GaussianCloud.load(from: nsData, to: options.coordinateSystem)
        return GaussianCloudData(from: cloud)
    } catch {
        return GaussianCloudData()
    }
}

/// Free memory allocated by saveSPZC
public func freeSPZData(data: UnsafeMutablePointer<UInt8>?) {
    data?.deallocate()
}