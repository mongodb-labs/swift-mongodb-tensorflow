import MongoSwift
import TensorFlow

private enum Query {
    case find(filter: Document?, opts: FindOptions?)
    case aggregate(pipeline: [Document])

    fileprivate func execute(on collection: MongoCollection<Document>) throws -> MongoCursor<Document> {
        switch self {
        case let .find(filter, opts):
            if let filter = filter {
                return try collection.find(filter, options: opts)
            }
            return try collection.find(options: opts)
        case let .aggregate(pipeline):
            return try collection.aggregate(pipeline)
        }
    }
}

/// Dataset backed by a lazily evaluated cursor. Each iteration yields a `TensorGroup` mapped from the first generic
/// type to the second.
public struct MongoDataset<M: TensorGroupMapping & Codable, T>: Sequence where M.Group == T {
    public typealias Iterator = MongoDatasetIterator<M, T>
    public typealias Element = T

    private let client: MongoClient
    private let collection: MongoCollection<Document>
    private var query: Query
    private let groupFactory: () -> T
    private var batchSize: Int = 1

    /**
     * Initialize a `MongoDataset` from the result of the provided MongoDB aggregation pipeline.
     *
     * - Parameters:
     *   - uri: Optional, the MongoDB connection string designating which MongoDB instance to connect to. Defaults
     *          to localhost:27017.
     *   - db: The name of the database to perform the aggregation against.
     *   - collection: The name of the collection to perform the aggregation against.
     *   - groupFactory: A closure that can produce an empty instance of the `TensorGroup` to be populated.
     * - SeeAlso: https://docs.mongodb.com/manual/aggregation/
     */
    public init(uri: String? = nil,
                db: String,
                collection: String,
                pipeline: [Document],
                groupFactory: @escaping () -> T) throws {
        if let uri = uri {
            self.client = try MongoClient(uri)
        } else {
            self.client = try MongoClient()
        }
        self.collection = client.db(db).collection(collection)
        self.query = .aggregate(pipeline: pipeline)
        self.groupFactory = groupFactory
    }

    /**
     * Initialize a `MongoDataset` from the given MongoDB collection, optionally providing a filter, projection, or limit
     * on the number of results.
     *
     * - Parameters:
     *   - uri: Optional, the MongoDB connection string designating which MongoDB instance to connect to. Defaults
     *          to localhost:27017.
     *   - db: The name of the database to containing the dataset collection.
     *   - collection: The name of the collection containing the dataset.
     *   - filter: Optional, a filter that documents must match to be included in the dataset.
     *   - projection: Optional, a document specifying which fields should be included in the retrieved documents.
     *   - limit: Optional, a limit on the number of documents included in the dataset.
     *   - groupFactory: A closure that can produce an empty instance of the `TensorGroup` to be populated.
     */
    public init(uri: String? = nil,
                db: String,
                collection: String,
                filter: Document? = nil,
                projection: Document? = nil,
                limit: Int64? = nil,
                groupFactory: @escaping () -> T) throws {
        if let uri = uri {
            self.client = try MongoClient(uri)
        } else {
            self.client = try MongoClient()
        }

        self.collection = self.client.db(db).collection(collection)
        let opts = FindOptions(limit: limit, projection: projection)
        self.query = .find(filter: filter, opts: opts)
        self.groupFactory = groupFactory
    }

    private init(copying dataset: MongoDataset<M, T>) {
        self.client = dataset.client
        self.collection = dataset.collection
        self.query = dataset.query
        self.groupFactory = dataset.groupFactory
    }

    /// Returns a copy of this dataset that populates the `TensorGroup`s yielded by iteration with batches of tensors
    /// from the given dataset.
    public func batched(_ batchSize: Int) throws -> MongoDataset<M, T> {
        var other = MongoDataset(copying: self)
        other.batchSize = batchSize
        return other
    }

    public func makeIterator() -> Iterator {
        do {
            let cursor = try self.query.execute(on: self.collection)
            return Iterator(wrapping: cursor, batchSize: self.batchSize, factory: self.groupFactory)
        } catch {
            fatalError("Error executing query: \(error)")
        }
    }
}

/// Iterator for a `MongoDataset` backed by a lazily evaluated cursor. Each iteration yields a batch of the dataset.
public struct MongoDatasetIterator<C: TensorGroupMapping & Codable, T>: IteratorProtocol where C.Group == T {
    private let cursor: MongoCursor<Document>
    private let batchSize: Int
    private let groupFactory: () -> T
    private var exhausted: Bool
    private var otherError: Error?

    /// The error that occurred while iterating the underlying cursor, if one exists.
    public var cursorError: Error? {
        if let cursorErr = self.cursor.error {
            return cursorErr
        }
        return self.otherError
    }

    internal init(wrapping cursor: MongoCursor<Document>, batchSize: Int, factory: @escaping () -> T) {
        self.cursor = cursor
        self.batchSize = batchSize
        self.exhausted = false
        self.groupFactory = factory
    }

    public mutating func next() -> T? {
        guard !exhausted else {
            return nil
        }

        let decoder = BSONDecoder()
        var elements: [C] = []
        for _ in 0 ..< self.batchSize {
            guard let record = self.cursor.next() else {
                self.exhausted = true
                break
            }

            do {
                let element = try decoder.decode(C.self, from: record)
                elements.append(element)
            } catch {
                self.otherError = error
                break
            }
        }

        guard !elements.isEmpty else {
            return nil
        }

        var group = self.groupFactory()

        do {
            try group.populate(mappings: elements)
        } catch {
            self.otherError = error
            return nil
        }

        return group
    }
}
