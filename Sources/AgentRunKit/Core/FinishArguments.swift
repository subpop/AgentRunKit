import Foundation

struct FinishArguments: Codable, Sendable {
    let content: String
    let reason: String?
}
