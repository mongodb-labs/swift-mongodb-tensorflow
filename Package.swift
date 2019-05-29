// swift-tools-version:5.0
import PackageDescription
let package = Package(
    name: "TFMongoSwift",
    products: [
        .library(name: "TFMongoSwift", targets: ["TFMongoSwift"])
    ],
    dependencies: [
        .package(url: "https://github.com/mongodb/mongo-swift-driver.git", .branch("master"))
    ],
    targets: [
        .target(name: "TFMongoSwift", dependencies: ["MongoSwift"]),
    ]
)
