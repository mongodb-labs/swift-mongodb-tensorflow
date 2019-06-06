import MongoSwift
import TensorFlow

public enum TFMongoSwiftError: Error {
    case dataCorrupt
}

private enum Query {
    case find(filter: Document?, opts: FindOptions)
    case aggregate(pipeline: [Document], opts: AggregateOptions)

    fileprivate var batchSize: Int32 {
        switch self {
        case let .find(_, opts):
            // this value is always set
            // swiftlint:disable:next force_unwrapping
            return opts.batchSize!
        case let .aggregate(_, opts):
            // this value is always set
            // swiftlint:disable:next force_unwrapping
            return opts.batchSize!
        }
    }

    fileprivate func execute(on collection: MongoCollection<Document>) throws -> MongoCursor<Document> {
        switch self {
        case let .find(filter, opts):
            if let filter = filter {
                return try collection.find(filter, options: opts)
            }
            return try collection.find(options: opts)
        case let .aggregate(pipeline, opts):
            return try collection.aggregate(pipeline, options: opts)
        }
    }
}

/// Dataset backed by a lazily evaluated cursor. Each batch makes a round trip to the server. Each iteration of the
/// iterator yields a `TensorGroup` converted from the first generic type to the second.
public struct MongoDataset<C: Codable, T: TensorGroup & InitializableFromSequence>: Sequence where T.SequenceType == C {
    public typealias Iterator = MongoDatasetIterator<C, T>
    public typealias Element = T

    private let client: MongoClient
    private let collection: MongoCollection<Document>
    private var query: Query

    private var batchSize: Int32 {
        get {
            return self.query.batchSize
        }
        set(newSize) {
            switch self.query {
            case let .find(filter, opts):
                let newOpts = FindOptions(batchSize: newSize, limit: opts.limit, projection: opts.projection)
                self.query = .find(filter: filter, opts: newOpts)
            case let .aggregate(pipeline, _):
                let newOpts = AggregateOptions(batchSize: newSize)
                self.query = .aggregate(pipeline: pipeline, opts: newOpts)
            }
        }
    }

    /// Initialize a `Dataset` from the result of the provided MongoDB aggregation pipeline.
    /// - SeeAlso: https://docs.mongodb.com/manual/aggregation/
    public init(uri: String? = nil, db: String, collection: String, pipeline: [Document]) throws {
        if let uri = uri {
            self.client = try MongoClient(uri)
        } else {
            self.client = try MongoClient()
        }
        self.collection = client.db(db).collection(collection)
        self.query = .aggregate(pipeline: pipeline, opts: AggregateOptions(batchSize: 1))
    }

    /// Initialize a `Dataset` from the given MongoDB collection, optionally providing a filter, projection, or limit
    /// on the number of results.
    public init(uri: String? = nil,
                db: String,
                collection: String,
                filter: Document? = nil,
                projection: Document? = nil,
                limit: Int64? = nil) throws {
        if let uri = uri {
            self.client = try MongoClient(uri)
        } else {
            self.client = try MongoClient()
        }

        self.collection = self.client.db(db).collection(collection)
        let opts = FindOptions(batchSize: 1, limit: limit, projection: projection)
        self.query = .find(filter: filter, opts: opts)
    }

    private init(copying dataset: MongoDataset<C, T>) {
        self.client = dataset.client
        self.collection = dataset.collection
        self.query = dataset.query
    }

    /// Returns a copy of this dataset that populates the `TensorGroup`s yielded by iteration with batches of tensors
    /// from the given dataset.
    public func batched(_ batchSize: Int32) throws -> MongoDataset<C, T> {
        var other = MongoDataset(copying: self)
        other.batchSize = batchSize
        return other
    }

    public func makeIterator() -> Iterator {
        guard let cursor = try? self.query.execute(on: self.collection) else {
            fatalError("couldn't get cursor")
        }
        return Iterator(wrapping: cursor, batchSize: self.batchSize)
    }
}

/// Iterator for a `MongoDataset` backed by a lazily evaluated cursor. Each iteration makes constitutes a round trip to
/// the server. The values are converted to their `TensorGroup` representation and combined into a single group.
public struct MongoDatasetIterator<C: Codable, T: TensorGroup & InitializableFromSequence>: IteratorProtocol
        where T.SequenceType == C {
    private let cursor: MongoCursor<Document>
    private let batchSize: Int32

    internal init(wrapping cursor: MongoCursor<Document>, batchSize: Int32) {
        self.cursor = cursor
        self.batchSize = batchSize
    }

    public mutating func next() -> T? {
        let decoder = BSONDecoder()
        var elements: [C] = []
        for _ in 0 ..< self.batchSize {
            guard let record = self.cursor.next() else {
                if let error = self.cursor.error {
                    if case UserError.logicError(_) = error {
                        continue
                    } else {
                        print("failed iterating cursor: \(error)")
                    }
                }
                break
            }
            guard let element = try? decoder.decode(C.self, from: record) else {
                print("failed decoding from \(record)")
                break
            }
            elements.append(element)
        }

        guard !elements.isEmpty else {
            return nil
        }

        return T(from: elements)
    }
}

extension Dataset {
    /// Initialize a `Dataset` from the result of a MongoDB query.
    /// An optional "transform" function can also be passed in that mutates the entire dataset before it is batched
    /// (e.g. for normalization).
    public init<T: Codable>(from cursor: MongoCursor<T>, transform: ((inout Element) -> Void)? = nil)
            where Element: InitializableFromSequence, Element.SequenceType == T {
        var elements = Element(from: Array(cursor))

        if let f = transform {
            f(&elements)
        }

        self.init(elements: elements)
    }

    /// Initialize a `Dataset` from the result of the provided MongoDB aggregation pipeline.
    /// An optional "transform" function can also be passed in that mutates the entire dataset before it is batched
    /// (e.g. for normalization).
    /// - SeeAlso: https://docs.mongodb.com/manual/aggregation/
    public init<C: Codable>(uri: String? = nil,
                            db: String,
                            collection: String,
                            pipeline: [Document],
                            outputType: C.Type,
                            transform: ((inout Element) -> Void)? = nil) throws
            where Element: InitializableFromSequence, Element.SequenceType == C {
        let client: MongoClient
        if let uri = uri {
            client = try MongoClient(uri)
        } else {
            client = try MongoClient()
        }
        let coll = client.db(db).collection(collection)
        let cursor = try coll.aggregate(pipeline)

        let decoder = BSONDecoder()
        var elements = Element(from: try cursor.map { try decoder.decode(C.self, from: $0) })

        if let f = transform {
            f(&elements)
        }

        self.init(elements: elements)
    }

    /// Initialize a `Dataset` from the given MongoDB collection, optionally providing a filter, projection, or limit
    /// on the number of results.
    /// An optional "transform" function can also be passed in that mutates the entire dataset before it is batched
    /// (e.g. for normalization).
    public init<C: Codable>(uri: String? = nil,
                            db: String,
                            collection: String,
                            filter: Document = [:],
                            projection: Document? = nil,
                            limit: Int64? = nil,
                            collectionType: C.Type,
                            transform: ((inout Element) -> Void)? = nil) throws
            where Element: InitializableFromSequence, Element.SequenceType == C {
        let client: MongoClient
        if let uri = uri {
            client = try MongoClient(uri)
        } else {
            client = try MongoClient()
        }
        let coll = client.db(db).collection(collection, withType: C.self)
        let cursor = try coll.find(filter, options: FindOptions(limit: limit, projection: projection))
        self.init(from: cursor, transform: transform)
    }
}
