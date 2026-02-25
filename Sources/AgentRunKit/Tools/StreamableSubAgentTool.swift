import Foundation

protocol StreamableSubAgentTool<Context>: AnyTool {
    func executeStreaming(
        toolCallId: String,
        arguments: Data,
        context: Context,
        eventHandler: @Sendable (StreamEvent) -> Void
    ) async throws -> ToolResult
}
