@testable import AgentRunKit
import Foundation
import Testing

private let apiKey = ProcessInfo.processInfo.environment["OPENROUTER_API_KEY"] ?? ""
private let hasAPIKey = !apiKey.isEmpty
private let model = ProcessInfo.processInfo.environment["SMOKE_OPENROUTER_MODEL"] ?? "google/gemini-3-flash-preview"

@Suite(.enabled(if: hasAPIKey, "Requires OPENROUTER_API_KEY environment variable"))
struct ContextManagementSmokeTests {
    let client = OpenAIClient(
        apiKey: apiKey,
        model: model,
        maxTokens: 1024,
        baseURL: OpenAIClient.openRouterBaseURL
    )

    let budgetClient = OpenAIClient(
        apiKey: apiKey,
        model: model,
        maxTokens: 1024,
        contextWindowSize: 500,
        baseURL: OpenAIClient.openRouterBaseURL
    )

    @Test func observationPruning() async throws {
        try await assertSmokeObservationPruning(client: budgetClient)
    }

    @Test func llmSummarization() async throws {
        try await assertSmokeLLMSummarization(client: budgetClient)
    }

    @Test func toolResultTruncation() async throws {
        try await assertSmokeToolResultTruncation(client: client)
    }

    @Test func maxMessages() async throws {
        try await assertSmokeMaxMessages(client: client)
    }

    @Test func budgetEvents() async throws {
        try await assertSmokeBudgetEvents(client: budgetClient)
    }

    @Test func iterationCompleted() async throws {
        try await assertSmokeIterationCompleted(client: client)
    }
}
