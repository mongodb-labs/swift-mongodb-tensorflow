import MongoSwift
import TensorFlow

public enum TFMongoSwiftError: Error {
    case dataCorrupt
}

/// Dataset backed by a lazily evaluated cursor. Each batch makes a round trip to the server. Each iteration of the
/// iterator yields a `TensorGroup` converted from the generic type associated with this Dataset.
public struct MongoDataset<T: TensorGroupConvertible & Codable>: Sequence where T.TensorType: Combinable {
    public typealias Iterator = MongoDatasetIterator<T>
    public typealias Element = T.TensorType

    private let client: MongoClient
    private let db: String
    private let collection: String
    private var batchSize: Int32

    public init(uri: String? = nil,
                db: String,
                collection: String,
                mapping type: T.Type,
                batchSize: Int32 = 32) throws {
        if let uri = uri {
            self.client = try MongoClient(uri)
        } else {
            self.client = try MongoClient()
        }
        self.db = db
        self.collection = collection
        self.batchSize = batchSize
    }

    public func batched(_ batchSize: Int32) throws -> MongoDataset<T> {
        return try MongoDataset(db: self.db, collection: self.collection, mapping: T.self, batchSize: batchSize)
    }

    public func makeIterator() -> Iterator {
        let coll = client.db(self.db).collection(self.collection, withType: T.self)
        guard let cursor = try? coll.find(options: FindOptions(batchSize: self.batchSize)) else {
            fatalError("error creating cursor")
        }
        return MongoDatasetIterator(wrapping: cursor, batchSize: self.batchSize)
    }
}

/// Iterator for a `MongoDataset` backed by a lazily evaluated cursor. Each iteration makes constitutes a round trip to
/// the server. The values are converted to their `TensorGroup` representation and combined into a single group.
public struct MongoDatasetIterator<T: TensorGroupConvertible & Codable>: IteratorProtocol
        where T.TensorType: Combinable {
    private let cursor: MongoCursor<T>
    private let batchSize: Int32

    internal init(wrapping cursor: MongoCursor<T>, batchSize: Int32) {
        self.cursor = cursor
        self.batchSize = batchSize
    }

    public mutating func next() -> T.TensorType? {
        var tensor: T.TensorType?
        for _ in 0 ..< self.batchSize {
            guard let record = self.cursor.next() else {
                break
            }
            tensor = tensor?.combined(with: record.tensorValue) ?? record.tensorValue
        }
        return tensor
    }
}

extension Dataset {
    public init<T: TensorGroupConvertible & Codable>(from collection: MongoCollection<T>) throws
            where T.TensorType == Element, Element: Combinable {
        let cursor = try collection.find()
        let tensors = cursor.reduce(nil) { (cumul: T.TensorType?, sample: T) in
            cumul?.combined(with: sample.tensorValue) ?? sample.tensorValue
        }

        guard let t = tensors else {
            throw TFMongoSwiftError.dataCorrupt
        }

        self.init(elements: t)
    }

    /// Gets a TensorFlow `Dataset` by converting values read from the given namespace to a `TensorGroup` as per
    /// the generic `TensorGroupConvertible` type.
    public init<T: TensorGroupConvertible & Codable>(uri: String? = nil,
                                                     db: String,
                                                     collection: String,
                                                     mapping: T.Type) throws
            where T.TensorType == Element, Element: Combinable {
        let client: MongoClient
        if let uri = uri {
            client = try MongoClient(uri)
        } else {
            client = try MongoClient()
        }
        let coll = client.db(db).collection(collection, withType: T.self)

        try self.init(from: coll)
    }
}
