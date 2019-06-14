public enum TFMongoSwiftError: Error {
    /// The mappings provided as part of a `TensorGroupMapping` did not all have the same shape/format.
    case nonUniformMappings

    /// No scalars were found for a particular `Tensor` on a `TensorGroup` being mapped to with the default
    /// `TensorGroupMapping` conformance.
    case noMatchingScalars(message: String)
}
