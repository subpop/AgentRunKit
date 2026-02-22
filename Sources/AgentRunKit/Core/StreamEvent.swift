import Foundation

public enum StreamEvent: Sendable, Equatable {
    case delta(String)
    case reasoningDelta(String)
    case toolCallStarted(name: String, id: String)
    case toolCallCompleted(id: String, name: String, result: ToolResult)
    case audioData(Data)
    case audioTranscript(String)
    case audioFinished(id: String, expiresAt: Int, data: Data)
    case finished(tokenUsage: TokenUsage, content: String?, reason: FinishReason?, history: [ChatMessage])
}
