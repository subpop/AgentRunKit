import Foundation

/// How the agent loop terminated.
public enum FinishReason: Sendable, Equatable, CustomStringConvertible {
    case completed
    case error
    case custom(String)

    public init(_ rawValue: String) {
        switch rawValue.lowercased() {
        case "completed": self = .completed
        case "error": self = .error
        default: self = .custom(rawValue)
        }
    }

    public var description: String {
        switch self {
        case .completed: "completed"
        case .error: "error"
        case let .custom(value): value
        }
    }
}

extension FinishReason: Codable {
    private enum CodingKeys: String, CodingKey {
        case type, value
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "completed": self = .completed
        case "error": self = .error
        case "custom": self = try .custom(container.decode(String.self, forKey: .value))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container,
                debugDescription: "Unknown FinishReason type: \(type)"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .completed: try container.encode("completed", forKey: .type)
        case .error: try container.encode("error", forKey: .type)
        case let .custom(value):
            try container.encode("custom", forKey: .type)
            try container.encode(value, forKey: .value)
        }
    }
}

public struct AgentResult: Sendable, Equatable {
    public let finishReason: FinishReason
    public let content: String
    public let totalTokenUsage: TokenUsage
    public let iterations: Int
    public let history: [ChatMessage]

    public init(
        finishReason: FinishReason,
        content: String,
        totalTokenUsage: TokenUsage,
        iterations: Int,
        history: [ChatMessage] = []
    ) {
        self.finishReason = finishReason
        self.content = content
        self.totalTokenUsage = totalTokenUsage
        self.iterations = iterations
        self.history = history
    }
}
