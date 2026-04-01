@testable import AgentRunKit
import Testing

private let promptTooLongError = AgentError.llmError(
    .httpError(statusCode: 400, body: "context_length_exceeded")
)

private struct ScriptedStreamClient: LLMClient {
    let deltas: [StreamDelta]
    let error: (any Error)?

    func generate(
        messages _: [ChatMessage],
        tools _: [ToolDefinition],
        responseFormat _: ResponseFormat?,
        requestContext _: RequestContext?
    ) async throws -> AssistantMessage {
        AssistantMessage(content: "")
    }

    nonisolated func stream(
        messages _: [ChatMessage],
        tools _: [ToolDefinition],
        requestContext _: RequestContext?
    ) -> AsyncThrowingStream<StreamDelta, Error> {
        let deltas = deltas
        let error = error
        let (stream, continuation) = AsyncThrowingStream<StreamDelta, Error>.makeStream()
        for delta in deltas {
            continuation.yield(delta)
        }
        continuation.finish(throwing: error)
        return stream
    }
}

struct StreamProcessorEmittedOutputTests {
    @Test
    func errorBeforeAnyDeltaSetsEmittedOutputFalse() async {
        let client = ScriptedStreamClient(deltas: [], error: promptTooLongError)
        let processor = StreamProcessor(client: client, toolDefinitions: [], policy: .chat)
        let (_, eventContinuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        var totalUsage = TokenUsage()
        var emittedOutput = true

        do {
            _ = try await processor.process(
                messages: [.user("Hi")],
                totalUsage: &totalUsage,
                emittedOutput: &emittedOutput,
                continuation: eventContinuation
            )
            Issue.record("Expected error")
        } catch {
            #expect(!emittedOutput)
        }
    }

    @Test
    func errorAfterContentDeltaSetsEmittedOutputTrue() async {
        let client = ScriptedStreamClient(
            deltas: [.content("hello")],
            error: promptTooLongError
        )
        let processor = StreamProcessor(client: client, toolDefinitions: [], policy: .chat)
        let (_, eventContinuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        var totalUsage = TokenUsage()
        var emittedOutput = false

        do {
            _ = try await processor.process(
                messages: [.user("Hi")],
                totalUsage: &totalUsage,
                emittedOutput: &emittedOutput,
                continuation: eventContinuation
            )
            Issue.record("Expected error")
        } catch {
            #expect(emittedOutput)
        }
    }

    @Test
    func toolCallStartUnderChatPolicySetsEmittedOutput() async {
        let client = ScriptedStreamClient(
            deltas: [.toolCallStart(index: 0, id: "call_1", name: "search")],
            error: promptTooLongError
        )
        let processor = StreamProcessor(client: client, toolDefinitions: [], policy: .chat)
        let (_, eventContinuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        var totalUsage = TokenUsage()
        var emittedOutput = false

        do {
            _ = try await processor.process(
                messages: [.user("Hi")],
                totalUsage: &totalUsage,
                emittedOutput: &emittedOutput,
                continuation: eventContinuation
            )
            Issue.record("Expected error")
        } catch {
            #expect(emittedOutput)
        }
    }

    @Test
    func terminalToolUnderAgentPolicyDoesNotSetEmittedOutput() async {
        let client = ScriptedStreamClient(
            deltas: [.toolCallStart(index: 0, id: "finish_1", name: "finish")],
            error: promptTooLongError
        )
        let processor = StreamProcessor(client: client, toolDefinitions: [], policy: .agent)
        let (_, eventContinuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream()
        var totalUsage = TokenUsage()
        var emittedOutput = true

        do {
            _ = try await processor.process(
                messages: [.user("Hi")],
                totalUsage: &totalUsage,
                emittedOutput: &emittedOutput,
                continuation: eventContinuation
            )
            Issue.record("Expected error")
        } catch {
            #expect(!emittedOutput)
        }
    }
}
