import Foundation

/// Errors that can occur during SPZ file operations
public enum SPZError: Error {
    case invalidHeader
    case unsupportedVersion
    case tooManyPoints
    case unsupportedSHDegree
    case readError
    case writeError
    case compressionError
    case decompressionError
    case invalidData
    case invalidFormat(String)
    case custom(String)
} 