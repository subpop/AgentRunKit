@testable import AgentRunKit
import Foundation
import Testing

private let apiKey = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? ""
private let hasAPIKey = !apiKey.isEmpty
private let model = ProcessInfo.processInfo.environment["SMOKE_ANTHROPIC_MODEL"] ?? "claude-sonnet-4-6"

@Suite(.enabled(if: hasAPIKey, "Requires ANTHROPIC_API_KEY environment variable"))
struct AnthropicSmokeTests {
    let client = AnthropicClient(apiKey: apiKey, model: model, maxTokens: 1024)

    @Test func basicGenerate() async throws {
        try await assertSmokeGenerate(client: client)
    }

    @Test func basicStream() async throws {
        try await assertSmokeStream(client: client)
    }

    @Test func toolCallRoundTrip() async throws {
        try await assertSmokeToolCall(client: client)
    }

    @Test func streamingToolCall() async throws {
        try await assertSmokeStreamingToolCall(client: client)
    }

    @Test func agentLoop() async throws {
        try await assertSmokeAgentLoop(client: client)
    }

    @Test func tokenUsagePresent() async throws {
        try await assertSmokeTokenUsage(client: client)
    }

    @Test func streamingAgentLoop() async throws {
        try await assertSmokeStreamingAgentLoop(client: client)
    }

    @Test func multiTurnConversation() async throws {
        try await assertSmokeMultiTurn(client: client)
    }

    @Test func streamingTokenUsage() async throws {
        try await assertSmokeStreamingTokenUsage(client: client)
    }

    @Test func reasoningStream() async throws {
        let thinkingClient = AnthropicClient(
            apiKey: apiKey,
            model: model,
            maxTokens: 16384,
            reasoningConfig: .budget(4096)
        )
        try await assertSmokeReasoningStream(client: thinkingClient)
    }

    @Test func reasoningGenerate() async throws {
        let thinkingClient = AnthropicClient(
            apiKey: apiKey,
            model: model,
            maxTokens: 16384,
            reasoningConfig: .budget(4096)
        )
        try await assertSmokeReasoningGenerate(client: thinkingClient)
    }
}
