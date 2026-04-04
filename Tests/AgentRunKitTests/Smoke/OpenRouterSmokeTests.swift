@testable import AgentRunKit
import Foundation
import Testing

private let apiKey = ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? ""
private let hasAPIKey = !apiKey.isEmpty
private let model = ProcessInfo.processInfo.environment["SMOKE_OPENROUTER_MODEL"] ?? "google/gemini-3-flash-preview"

@Suite(.enabled(if: hasAPIKey, "Requires OPENROUTER_API_KEY environment variable"))
struct OpenRouterSmokeTests {
    let client = OpenAIClient(
        apiKey: apiKey,
        model: model,
        maxTokens: 1024,
        baseURL: OpenAIClient.openRouterBaseURL
    )

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

    @Test func structuredOutput() async throws {
        try await assertSmokeStructuredOutput(client: client)
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

    @Test func chatStreamWithTools() async throws {
        try await assertSmokeChatStreamWithTools(client: client)
    }

    @Test func budgetHistoryIntegrity() async throws {
        let budgetClient = OpenAIClient(
            apiKey: apiKey,
            model: model,
            maxTokens: 1024,
            contextWindowSize: 100,
            baseURL: OpenAIClient.openRouterBaseURL
        )
        try await assertSmokeBudgetHistoryIntegrity(client: budgetClient)
    }

    @Test func nestedStructuredOutput() async throws {
        try await assertSmokeNestedStructuredOutput(client: client)
    }

    @Test func approvalGate() async throws {
        try await assertSmokeApprovalGate(client: client)
    }

    @Test func approvalDenial() async throws {
        try await assertSmokeApprovalDenial(client: client)
    }

    @Test func streamingApproval() async throws {
        try await assertSmokeStreamingApproval(client: client)
    }
}

private let reasoningModel = ProcessInfo.processInfo.environment["SMOKE_OPENROUTER_REASONING_MODEL"] ?? ""
private let hasReasoningModel = !reasoningModel.isEmpty

@Suite(.enabled(if: hasAPIKey && hasReasoningModel,
                "Requires OPENROUTER_API_KEY and SMOKE_OPENROUTER_REASONING_MODEL"))
struct OpenRouterReplayPolicySmokeTests {
    let client = OpenAIClient(
        apiKey: apiKey,
        model: reasoningModel,
        maxTokens: 4096,
        baseURL: OpenAIClient.openRouterBaseURL,
        reasoningConfig: .high,
        assistantReplayProfile: .openRouterReasoningDetails
    )

    @Test func multiTurnReasoningDetailsReplay() async throws {
        let turn1Messages: [ChatMessage] = [
            .system("You are a helpful assistant. Think step by step."),
            .user("What is 7 * 13? Show your work."),
        ]

        let turn1 = try await client.generate(messages: turn1Messages, tools: [])
        #expect(!turn1.content.isEmpty)
        try #require(
            turn1.reasoningDetails != nil,
            "Model must return reasoning_details to exercise the replay contract"
        )

        var turn2Messages = turn1Messages
        turn2Messages.append(.assistant(turn1))
        turn2Messages.append(.user("Now double that result."))

        let turn2 = try await client.generate(messages: turn2Messages, tools: [])
        #expect(!turn2.content.isEmpty)
    }
}
