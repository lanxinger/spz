# spz

`.spz` is a file format for compressed 3D gaussian splats. This repository contains a Swift
implementation for saving and loading data in the .spz format, fully compatible with the
[official C++ implementation](https://github.com/nianticlabs/spz).

spz encoded splats are typically around 10x smaller than the corresponding .ply files,
with minimal visual differences between the two.

## Swift Implementation

### Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/lanxinger/spz.git", from: "1.0.0")
]
```

Or in Xcode: File > Add Package Dependencies and enter the repository URL.

### Requirements

- iOS 13.0+ / macOS 11.0+
- Swift 5.7+
- zlib (system library, no additional dependencies)

### Usage

```swift
import SPZ

// Load from SPZ file
let cloud = try GaussianCloud.load(from: spzURL, to: .rub)

// Load from PLY file
let cloud = try GaussianCloud.loadFromPly(url: plyURL, to: .rub)

// Save to SPZ file
try cloud.save(to: outputURL, from: .rub)

// Save to PLY file
try cloud.saveToPly(url: outputURL, from: .rub)

// Access gaussian data
print("Points: \(cloud.numPoints)")
print("SH Degree: \(cloud.shDegree)")
// Arrays: cloud.positions, cloud.scales, cloud.rotations, cloud.alphas, cloud.colors, cloud.sh
```

### API

#### GaussianCloud

The main data structure representing a cloud of 3D gaussians:

```swift
public struct GaussianCloud {
    var numPoints: Int           // Number of gaussians
    var shDegree: Int            // Spherical harmonics degree (0-3)
    var antialiased: Bool        // Mip splatting flag
    var positions: [Float]       // XYZ positions (numPoints * 3)
    var scales: [Float]          // Log-scale XYZ (numPoints * 3)
    var rotations: [Float]       // XYZW quaternions (numPoints * 4)
    var alphas: [Float]          // Pre-sigmoid alpha (numPoints)
    var colors: [Float]          // SH DC components (numPoints * 3)
    var sh: [Float]              // Spherical harmonics coefficients
}
```

#### Loading

```swift
// Load from SPZ file with coordinate conversion
static func load(from url: URL, to: CoordinateSystem) throws -> GaussianCloud

// Load from SPZ Data
static func load(from data: Data, to: CoordinateSystem) throws -> GaussianCloud

// Load from PLY file
static func loadFromPly(url: URL, to: CoordinateSystem) throws -> GaussianCloud
```

#### Saving

```swift
// Save to SPZ file
func save(to url: URL, from: CoordinateSystem) throws

// Save to SPZ Data
func save(from: CoordinateSystem) throws -> Data

// Save to PLY file
func saveToPly(url: URL, from: CoordinateSystem) throws
```

#### Coordinate Systems

```swift
public enum CoordinateSystem {
    case unspecified  // No conversion
    case rub          // Right-Up-Back (OpenGL, three.js) - SPZ internal format
    case rdf          // Right-Down-Front (PLY format)
    case luf          // Left-Up-Front (GLB format)
    case ruf          // Right-Up-Front (Unity)
    // ... and more
}
```

### Command Line Tool

The package includes `spz-tool` for file conversion:

```bash
# Build the tool
swift build -c release

# Convert PLY to SPZ
.build/release/spz-tool convert input.ply output.spz

# Convert SPZ to PLY
.build/release/spz-tool convert input.spz output.ply

# Show file info
.build/release/spz-tool info input.spz
```

## Internals

### Coordinate System

SPZ stores data internally in an RUB coordinate system following the OpenGL and three.js
convention. This differs from other data formats such as PLY (which typically uses RDF), GLB (which
typically uses LUF), or Unity (which typically uses RUF). To aid with coordinate system conversions,
callers should specify the coordinate system their Gaussian Cloud data is represented in when saving
(the `from` parameter) and what coordinate system their rendering system uses when loading (the `to`
parameter). If the coordinate system is `.unspecified`, data will be saved and loaded without
conversion, which may harm interoperability.

## File Format

The .spz format is a gzipped stream of data consisting of a 16-byte header followed by the
gaussian data. This data is organized by attribute in the following order: positions,
alphas, colors, scales, rotations, spherical harmonics.

### Header

```c
struct PackedGaussiansHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t numPoints;
  uint8_t shDegree;
  uint8_t fractionalBits;
  uint8_t flags;
  uint8_t reserved;
};
```

All values are little-endian.

1. **magic**: This is always 0x5053474e
2. **version**: Currently, the only valid versions are 2 and 3
3. **numPoints**: The number of gaussians
4. **shDegree**: The degree of spherical harmonics. This must be between 0 and 3 (inclusive).
5. **fractionalBits**: The number of bits used to store the fractional part of coordinates in
   the fixed-point encoding.
6. **flags**: A bit field containing flags.
   - `0x1`: whether the splat was trained with [antialiasing](https://niujinshuchong.github.io/mip-splatting/).
7. **reserved**: Reserved for future use. Must be 0.

### Positions

Positions are represented as `(x, y, z)` coordinates, each as a 24-bit fixed point signed integer.
The number of fractional bits is determined by the `fractionalBits` field in the header.

### Scales

Scales are represented as `(x, y, z)` components, each represented as an 8-bit log-encoded integer.

### Rotation

In version 3, rotations are represented as the smallest three components of the normalized rotation quaternion, for optimal rotation accuracy.
The largest component can be derived from the others and is not stored. Its index is stored on 2 bits
and each of the smallest three components is encoded as a 10-bit signed integer.

In version 2, rotations are represented as the `(x, y, z)` components of the normalized rotation quaternion. The
`w` component can be derived from the others and is not stored. Each component is encoded as an
8-bit signed integer.

### Alphas

Alphas are represented as 8-bit unsigned integers.

### Colors

Colors are stored as `(r, g, b)` values, where each color component is represented as an
unsigned 8-bit integer.

### Spherical Harmonics

Depending on the degree of spherical harmonics for the splat, this can contain 0 (for degree 0),
9 (for degree 1), 24 (for degree 2), or 45 (for degree 3) coefficients per gaussian.

The coefficients for a gaussian are organized such that the color channel is the inner (faster
varying) axis, and the coefficient is the outer (slower varying) axis, i.e. for degree 1,
the order of the 9 values is:

```
sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, sh10_g, sh10_b, sh1p1_r, sh1p1_g, sh1p1_b
```

Each coefficient is represented as an 8-bit signed integer. Additional quantization can be performed
to attain a higher compression ratio. This library currently uses 5 bits of precision for degree 0
and 4 bits of precision for degrees 1 and 2, but this may be changed in the future without breaking
backwards compatibility.


## Other Implementations

- **C++**: See the [official Niantic Labs repository](https://github.com/nianticlabs/spz) for the reference C++ implementation
- **Python**: Python bindings are available in the [official repository](https://github.com/nianticlabs/spz) via nanobind
