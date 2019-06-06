import Foundation
import MongoSwift
import TensorFlow

/// This file defines protocols and structures that help facilitate mapping between data stored in MongoDB and
/// `TensorGroup`s.

/// A basic `TensorGroup` that consisting of a `Tensor` of features and a `Tensor` of labels.
public struct FLB<T: FeatureLabelMapping>: TensorGroup, InitializableFromSequence {
    public typealias SequenceType = T

    /// The features for this batch of data.
    public var features: Tensor<T.FeatureType>

    /// The labels for this batch of data.
    public var labels: Tensor<T.LabelType>

    public init<S: Sequence>(from elements: S) where S.Element == SequenceType {
        var featureScalars: [T.FeatureType] = []
        var labelScalars: [T.LabelType] = []

        for element in elements {
            featureScalars += element.features
            labelScalars.append(element.label)
        }

        var shape = T.featureShape
        shape[0] = labelScalars.count
        self.features = Tensor(shape: shape, scalars: featureScalars)
        self.labels = Tensor(labelScalars)
    }
}

/// Allows defining a direct mapping to a `FeatureLabelBatch`.
public protocol FeatureLabelMapping: Codable {
    associatedtype FeatureType: TensorFlowScalar
    associatedtype LabelType: TensorFlowScalar

    static var featureShape: TensorShape { get }
    var features: [FeatureType] { get }
    var label: LabelType { get }
}

/// Allows initializing a type from a sequence of another type. Required for mapping a collection's data to a
/// `TensorGroup`.
public protocol InitializableFromSequence {
    /// Can be initialized from a sequence of this type.
    associatedtype SequenceType

    /// Initialize from a sequence of the associated `SequenceType`.
    init<S: Sequence>(from sequence: S) where S.Element == SequenceType
}
