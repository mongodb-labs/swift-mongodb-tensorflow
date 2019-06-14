import Foundation
import MongoSwift
import TensorFlow

/// This file defines protocols and structures that help facilitate mapping between data stored in MongoDB and
/// `TensorGroup`s.

/// Enum specifying the possible maps from a path on a `TensorGroup` to a `Tensor`.
public enum TensorMap<G: TensorGroup> {
    /// A mapping of a `KeyPath` to a `Tensor<Bool>` to a `Tensor<Bool>`.
    case bool(WritableKeyPath<G, Tensor<Bool>>, Tensor<Bool>)

    /// A mapping of a `KeyPath` to a `Tensor<Int8>` to a `Tensor<Int8>`.
    case int8(WritableKeyPath<G, Tensor<Int8>>, Tensor<Int8>)

    /// A mapping of a `KeyPath` to a `Tensor<Int16>` to a `Tensor<Int16>`.
    case int16(WritableKeyPath<G, Tensor<Int16>>, Tensor<Int16>)

    /// A mapping of a `KeyPath` to a `Tensor<Int32>` to a `Tensor<Int32>`.
    case int32(WritableKeyPath<G, Tensor<Int32>>, Tensor<Int32>)

    /// A mapping of a `KeyPath` to a `Tensor<Int64>` to a `Tensor<Int64>`.
    case int64(WritableKeyPath<G, Tensor<Int64>>, Tensor<Int64>)

    /// A mapping of a `KeyPath` to a `Tensor<Double>` to a `Tensor<Double>`.
    case double(WritableKeyPath<G, Tensor<Double>>, Tensor<Double>)

    /// A mapping of a `KeyPath` to a `Tensor<Float>` to a `Tensor<Float>`.
    case float(WritableKeyPath<G, Tensor<Float>>, Tensor<Float>)

    /// A mapping of a `KeyPath` to a `Tensor<UInt8>` to a `Tensor<UInt8>`.
    case uint8(WritableKeyPath<G, Tensor<UInt8>>, Tensor<UInt8>)

    /// A mapping of a `KeyPath` to a `Tensor<UInt16>` to a `Tensor<UInt16>`.
    case uint16(WritableKeyPath<G, Tensor<UInt16>>, Tensor<UInt16>)

    /// A mapping of a `KeyPath` to a `Tensor<UInt32>` to a `Tensor<UInt32>`.
    case uint32(WritableKeyPath<G, Tensor<UInt32>>, Tensor<UInt32>)

    /// A mapping of a `KeyPath` to a `Tensor<UInt64>` to a `Tensor<UInt64>`.
    case uint64(WritableKeyPath<G, Tensor<UInt64>>, Tensor<UInt64>)

    // Gets the `Tensor` from this map if the path matches.
    // swiftlint:disable:next cyclomatic_complexity
    fileprivate func extractTensor<T: TensorFlowScalar>(at path: WritableKeyPath<G, Tensor<T>>) -> Tensor<T>? {
        switch T.self {
        case is Bool.Type:
            guard case let .bool(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Int8.Type:
            guard case let .int8(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Int16.Type:
            guard case let .int16(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Int32.Type:
            guard case let .int32(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Int64.Type:
            guard case let .int64(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Double.Type:
            guard case let .double(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is Float.Type:
            guard case let .float(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is UInt8.Type:
            guard case let .uint8(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is UInt16.Type:
            guard case let .uint16(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is UInt32.Type:
            guard case let .uint32(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        case is UInt64.Type:
            guard case let .uint64(mapPath, tensor) = self, path == mapPath else {
                return nil
            }
            return tensor as? Tensor<T>
        default:
            return nil
        }
    }
}

/// Defines a custom mapping of this object to an associated `TensorGroup`.
/// The default conformance of this protocol will map all the properties that conform to `TensorFlowScalar` to the first
/// `Tensor` of that type found on the associated `TensorGroup`.
public protocol TensorGroupMapping: KeyPathIterable {
    /// The `TensorGroup` this maps to.
    associatedtype Group: KeyPathIterable, TensorGroup

    /// The mapping from the output `TensorGroup`'s properties to custom values.
    var mapping: [TensorMap<Self.Group>]? { get }
}

extension TensorGroupMapping {
    /// The default mapping (nil). This means that all properties that conform to `TensorFlowScalar` will be mapped to
    /// the first `Tensor` of that type found on the associated `TensorGroup`.
    public var mapping: [TensorMap<Self.Group>]? { return nil }
}

extension TensorGroup where Self: KeyPathIterable {
    /// Concatenate the provided scalars to the `Tensor` found at the path. If overwrite is specified, it will replace
    /// the existing tensor with a new one constructed from the scalars.
    private mutating func concatenateTensors<T: TensorFlowScalar>(at path: WritableKeyPath<Self, Tensor<T>>,
                                                                  tensor: Tensor<T>,
                                                                  overwrite: Bool) throws {
        guard overwrite || tensor.shape.dimensions[0...] == self[keyPath: path].shape.dimensions[1...] else {
            throw TFMongoSwiftError.nonUniformMappings
        }

        let newTensor = tensor.reshaped(to: TensorShape([1] + tensor.shape.dimensions))

        if overwrite {
            self[keyPath: path] = newTensor
        } else {
            self[keyPath: path] = self[keyPath: path].concatenated(with: newTensor)
        }
    }

    /// Map all the scalars of type `T` from the array of `KeyPathIterable`s to the first writeable `Tensor<T>` field.
    private mutating func mapAllScalarsToFirst<T: TensorFlowScalar, K: KeyPathIterable>(type: T.Type,
                                                                                        elements: [K]) throws {
        var scalars: [T] = []
        if let path = self.allWritableKeyPaths(to: Tensor<T>.self).first {
            scalars += elements.flatMap { s in s.allKeyPaths(to: T.self).map { s[keyPath: $0] } }
            scalars += elements.flatMap { s in s.allKeyPaths(to: [T].self).flatMap { s[keyPath: $0] } }

            guard !scalars.isEmpty else {
                throw TFMongoSwiftError.noMatchingScalars(message: "Could not find any scalars of type \(T.self) to " +
                        "map.")
            }

            self[keyPath: path] = Tensor<T>(shape: [elements.count, scalars.count / elements.count], scalars: scalars)
        }
    }

    // swiftlint:disable force_unwrapping
    /// Overwrites the `Tensor` at the given path by extracting the `Tensor` from each of the maps and combining all the
    /// scalars. All of the maps MUST share the same path component.
    private mutating func assignTensor<S: TensorFlowScalar>(at path: WritableKeyPath<Self, Tensor<S>>,
                                                            maps: [TensorMap<Self>]) {
        guard !maps.isEmpty else {
            return
        }
        // The extraction is guaranteed to succeed due to the path precondition of invoking this method.
        let scalars = maps.flatMap { $0.extractTensor(at: path)!.scalars }
        // Maps is guaranteed to be non-empty, the extraction should also succeed here for the same reason as the
        // previous.
        let shape = TensorShape([maps.count] + maps.first!.extractTensor(at: path)!.shape.dimensions)
        self[keyPath: path] = Tensor<S>(shape: shape, scalars: scalars)
    }
    // swiftlint:enable force_unwrapping

    // swiftlint:disable cyclomatic_complexity
    /// Populate this `TensorGroup` from the provided array of `TensorGroupMapping`s.
    public mutating func populate<M: TensorGroupMapping>(mappings: [M]) throws where M.Group == Self {
        guard !mappings.isEmpty else {
            return
        }

        guard let first = mappings.first?.mapping else {
            // map each tensor to values of the same scalar type
            try self.mapAllScalarsToFirst(type: Double.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Float.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Int8.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Int16.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Int32.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Int64.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: UInt8.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: UInt16.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: UInt32.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: UInt64.self, elements: mappings)
            try self.mapAllScalarsToFirst(type: Bool.self, elements: mappings)
            return
        }

        for (i, map) in first.enumerated() {
            let allMaps: [TensorMap<Self>] = try mappings.compactMap {
                guard let mapping = $0.mapping, mapping.count > i else {
                    throw TFMongoSwiftError.nonUniformMappings
                }
                return mapping[i]
            }

            switch map {
            case let .bool(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .int8(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .int16(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .int32(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .int64(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .double(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .float(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .uint8(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .uint16(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .uint32(path, _):
                self.assignTensor(at: path, maps: allMaps)
            case let .uint64(path, _):
                self.assignTensor(at: path, maps: allMaps)
            }
        }
    }
    // swiftlint:enable cyclomatic_complexity

    /// Returns a copy of this `TensorGroup` populated according to the given array of `TensorGroupMapping`s.
    public func populated<M: TensorGroupMapping>(mappings: [M]) throws -> Self where M.Group == Self {
        var copy = self
        try copy.populate(mappings: mappings)
        return copy
    }

    /// Returns a copy of this `TensorGroup` populated from the given MongoDB collection according to the provided
    /// `TensorGroupMapping`.
    public func populated<M: TensorGroupMapping & Codable>(uri: String? = nil,
                                                           db: String,
                                                           collection: String,
                                                           filter: Document = [:],
                                                           projection: Document? = nil,
                                                           limit: Int64? = nil,
                                                           mapping: M.Type) throws -> Self where M.Group == Self {
        var copy = self
        try copy.populate(uri: uri,
                          db: db,
                          collection: collection,
                          filter: filter,
                          projection: projection,
                          limit: limit,
                          mapping: M.self)
        return copy
    }

    /// Populate this `TensorGroup` from the given MongoDB collection according to the provided `TensorGroupMapping`.
    public mutating func populate<M: TensorGroupMapping & Codable>(uri: String? = nil,
                                                                   db: String,
                                                                   collection: String,
                                                                   filter: Document = [:],
                                                                   projection: Document? = nil,
                                                                   limit: Int64? = nil,
                                                                   mapping: M.Type) throws where M.Group == Self {
        let client: MongoClient
        if let uri = uri {
            client = try MongoClient(uri)
        } else {
            client = try MongoClient()
        }

        let options = FindOptions(limit: limit, projection: projection)
        let data = try Array(client.db(db).collection(collection, withType: M.self).find(filter, options: options))
        try self.populate(mappings: data)
    }

    /// Returns a copy of this `TensorGroup` populated from the results of the given MongoDB aggregation query on the
    /// given namespace according to the provided `TensorGroupMapping`.
    /// `TensorGroupMapping`.
    public func populated<M: TensorGroupMapping & Codable>(uri: String? = nil,
                                                           db: String,
                                                           collection: String,
                                                           pipeline: [Document],
                                                           mapping: M.Type) throws -> Self where M.Group == Self {
        var copy = self
        try copy.populate(uri: uri,
                          db: db,
                          collection: collection,
                          pipeline: pipeline,
                          mapping: M.self)
        return copy
    }

    /// Populate this `TensorGroup` from the results of the given MongoDB aggregation query on the given namespace
    /// according to the provided `TensorGroupMapping`.
    /// - SeeAlso: https://docs.mongodb.com/manual/aggregation/
    public mutating func populate<M: TensorGroupMapping & Codable>(uri: String? = nil,
                                                                   db: String,
                                                                   collection: String,
                                                                   pipeline: [Document],
                                                                   mapping: M.Type) throws where M.Group == Self {
        let client: MongoClient
        if let uri = uri {
            client = try MongoClient(uri)
        } else {
            client = try MongoClient()
        }

        let data = try client
                .db(db)
                .collection(collection)
                .aggregate(pipeline)
                .map { try BSONDecoder().decode(M.self, from: $0) }
        try self.populate(mappings: data)
    }
}
