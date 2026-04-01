import Foundation

enum RunRequestMode: Equatable {
    case auto
    case forceFullRequest
}

protocol HistoryRewriteAwareClient: LLMClient {
    func generate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        responseFormat: ResponseFormat?,
        requestContext: RequestContext?,
        requestMode: RunRequestMode
    ) async throws -> AssistantMessage

    func stream(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        requestContext: RequestContext?,
        requestMode: RunRequestMode
    ) -> AsyncThrowingStream<StreamDelta, Error>
}

extension HistoryRewriteAwareClient {
    func stream(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        requestContext: RequestContext?,
        requestMode _: RunRequestMode
    ) -> AsyncThrowingStream<StreamDelta, Error> {
        stream(messages: messages, tools: tools, requestContext: requestContext)
    }
}

extension LLMClient {
    func generateForRun(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        responseFormat: ResponseFormat?,
        requestContext: RequestContext?,
        requestMode: RunRequestMode
    ) async throws -> AssistantMessage {
        if let capableClient = self as? any HistoryRewriteAwareClient {
            return try await capableClient.generate(
                messages: messages,
                tools: tools,
                responseFormat: responseFormat,
                requestContext: requestContext,
                requestMode: requestMode
            )
        }
        return try await generate(
            messages: messages,
            tools: tools,
            responseFormat: responseFormat,
            requestContext: requestContext
        )
    }

    func streamForRun(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        requestContext: RequestContext?,
        requestMode: RunRequestMode
    ) -> AsyncThrowingStream<StreamDelta, Error> {
        if let capableClient = self as? any HistoryRewriteAwareClient {
            return capableClient.stream(
                messages: messages,
                tools: tools,
                requestContext: requestContext,
                requestMode: requestMode
            )
        }
        return stream(messages: messages, tools: tools, requestContext: requestContext)
    }
}
