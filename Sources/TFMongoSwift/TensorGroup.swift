import Foundation
import MongoSwift
import TensorFlow

/// This file defines protocols that help facilitate mapping between data stored in MongoDB and `TensorGroup`s.

/// Allows a given `TensorGroup` to be combined (e.g. adding rows) with another `TensorGroup` of the same type.
public protocol Combinable where Self: TensorGroup {
    /// Returns a new `TensorGroup` of the same type as `self` which is the result of combining `self` with the supplied
    /// `TensorGroup`.
    func combined(with other: Self) -> Self
}

/// A type conforming to `TensorGroup` and `Combinable` that contains two tensors: one for features and one for labels.
/// Can be used as a generic batch type on a Dataset for a wide variety of applications.
public struct FeatureLabelBatch<FeatureType: TensorFlowScalar, LabelType: TensorFlowScalar>: TensorGroup, Combinable {
    public let features: Tensor<FeatureType>
    public let labels: Tensor<LabelType>

    public func combined(with other: FeatureLabelBatch<FeatureType, LabelType>)
                    -> FeatureLabelBatch<FeatureType, LabelType> {
        return FeatureLabelBatch(features: self.features.concatenated(with: other.features),
                                 labels: self.labels.concatenated(with: other.labels))
    }
}

/// Allows a type to be converted to a given `TensorGroup`.
public protocol TensorGroupConvertible {
    associatedtype TensorType: TensorGroup

    var tensorValue: TensorType { get }
}

/// Allows defining a direct mapping to a `FeatureLabelBatch`.
public protocol FeatureLabelMapping: TensorGroupConvertible, Codable
        where Self.TensorType == FeatureLabelBatch<FeatureType, LabelType> {
    associatedtype FeatureType: TensorFlowScalar
    associatedtype LabelType: TensorFlowScalar

    static var NUM_FEATURES: Int { get }
    var features: [FeatureType] { get }
    var label: LabelType { get }
}

extension FeatureLabelMapping {
    public var tensorValue: TensorType {
        let features = self.features
        let featuresTensor = Tensor<FeatureType>(shape: [1, features.count], scalars: features)
        let labels = Tensor<LabelType>([self.label])
        return FeatureLabelBatch(features: featuresTensor, labels: labels)
    }
}
