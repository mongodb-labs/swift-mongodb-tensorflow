import Foundation
import TensorFlow
import TFMongoSwift

// Example based off of https://www.tensorflow.org/swift/tutorials/model_training_walkthrough
// To run this example, use mongoimport to import http://download.tensorflow.org/data/iris_training.csv to a MongoDB instance.
//
// The example is currently written to assume field names "sepalLength", "sepalWidth", "petalLength", "petalWidth", and "irisLabel".
// It also assumes that the database is a standalone running at localhost:27017, and the data is in tf.iris_train and tf.iris_test.
// Of course, these can all be configured by changing the example below.

/// TensorFlow model
struct IrisModel: Layer {
    typealias Input = Tensor<Double>
    typealias Output = Tensor<Double>

    var layer2 = Dense<Double>(inputSize: 4, outputSize: 10, activation: relu)
    var layer3 = Dense<Double>(inputSize: 10, outputSize: IrisLabel.allCases.count)

    @differentiable
    func call(_ input: Tensor<Double>) -> Tensor<Double> {
        return input.sequenced(through: layer2, layer3)
    }
}

/// `TensorGroup` used for training in TensorFlow.
public struct IrisBatch: TensorGroup, KeyPathIterable {
    public var features: Tensor<Double>
    public var labels: Tensor<Int32>
}

/// Classifications of flowers.
public enum IrisLabel: Int32, Codable, CaseIterable {
    case setosa = 0
    case versicolor
    case virginica
}

/// `Codable` model of data in MongoDB.
public struct IrisMapping: Codable {
    let sepalLength: Double
    let sepalWidth: Double
    let petalLength: Double
    let petalWidth: Double
    let irisLabel: IrisLabel
}

/// Explicit mapping from a document in MongoDB to a `TensorGroup`.
extension IrisMapping: TensorGroupMapping {
    public typealias Group = IrisBatch

    public var mapping: [TensorMap<IrisBatch>]? {
        return [
            .int32(\IrisBatch.labels, Tensor<Int32>(self.irisLabel.rawValue)),
            .double(\IrisBatch.features, Tensor<Double>([self.sepalLength, self.sepalWidth, self.petalLength, self.petalWidth]))
        ]
    }
}

/// The "synthesized" model + mapping.
public struct IrisSynth: Codable, TensorGroupMapping {
    public typealias Group = IrisBatch

    let sepalLength: Double
    let sepalWidth: Double
    let petalLength: Double
    let petalWidth: Double
    let irisLabel: Int32
}

func iris() throws {
    let batchSize = 32

    let group = try IrisBatch(features: Tensor<Double>.zero, labels: Tensor<Int32>.zero)
            .populated(db: "tf", collection: "iris_train", mapping: IrisMapping.self)

    let dataset = Dataset(elements: group).batched(batchSize)

    var model = IrisModel()
    let optimizer: SGD<IrisModel> = SGD(for: model, learningRate: 0.01)

    let numEpochs = 250
    for epoch in 0..<numEpochs {
        var nPredictions = 0
        var correct: Int32 = 0

        print("running epoch \(epoch)")
        for (_, batch) in dataset.enumerated() {
            let (_, grads) = model.valueWithGradient { (model: IrisModel) -> IrisModel.Output in
                let logits = model.call(batch.features)
                return softmaxCrossEntropy(logits: logits, labels: batch.labels)
            }
            optimizer.update(&model.allDifferentiableVariables, along: grads)

            let logits = model.call(batch.features).argmax(squeezingAxis: 1)
            correct += Tensor<Int32>(logits .== batch.labels).sum().scalarized()
            nPredictions += logits.scalarCount
        }
        print("Epoch \(epoch): Accuracy: \(Float(correct) / Float(nPredictions))")
    }

    let testSet = try MongoDataset<IrisSynth, IrisBatch>(db: "tf",
                                                 collection: "iris_test",
                                                 groupFactory: { IrisBatch(features: Tensor<Double>.zero, labels: Tensor<Int32>.zero) })
    .batched(batchSize)

    var correct: Int32 = 0
    var nPredictions: Int = 0
    for batch in testSet {
        let batchImages = batch.features
        let batchLabels = batch.labels.flattened()

        let predictions = model.call(batchImages).argmax(squeezingAxis: 1)
        nPredictions += predictions.scalarCount
        correct += Tensor<Int32>(predictions .== batchLabels).sum().scalarized()
    }
    print("Test accuracy: \(Float(correct) / Float(nPredictions))")
}

let startTime = CFAbsoluteTimeGetCurrent()
try iris()
let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
print("Time: \(timeElapsed)")
