@testable import AgentRunKit
import Foundation
import Testing

private let vertexProjectID = ProcessInfo.processInfo.environment["VERTEX_PROJECT_ID"] ?? ""
private let vertexLocation = ProcessInfo.processInfo.environment["VERTEX_LOCATION"] ?? ""
private let anthropicModel = ProcessInfo.processInfo.environment["SMOKE_VERTEX_ANTHROPIC_MODEL"] ?? ""
private let hasVertexAnthropicConfig =
    !vertexProjectID.isEmpty
        && !vertexLocation.isEmpty
        && !anthropicModel.isEmpty
        && GoogleAuthService.credentialsAvailable()

@Suite(.enabled(
    if: hasVertexAnthropicConfig,
    "Requires VERTEX_PROJECT_ID, VERTEX_LOCATION, SMOKE_VERTEX_ANTHROPIC_MODEL, and Google ADC credentials"
))
struct VertexAnthropicSmokeTests {
    private func makeClient() throws -> VertexAnthropicClient {
        try VertexAnthropicClient(
            projectID: vertexProjectID,
            location: vertexLocation,
            model: anthropicModel,
            authService: GoogleAuthService(),
            maxTokens: 1024
        )
    }

    private func makeBudgetClient() throws -> VertexAnthropicClient {
        try VertexAnthropicClient(
            projectID: vertexProjectID,
            location: vertexLocation,
            model: anthropicModel,
            authService: GoogleAuthService(),
            maxTokens: 1024,
            contextWindowSize: 100
        )
    }

    @Test
    func budgetHistoryIntegrity() async throws {
        try await assertSmokeBudgetHistoryIntegrity(client: makeBudgetClient())
    }

    @Test
    func continuityReplay() async throws {
        try await assertSmokeAnthropicContinuityReplay(client: makeClient())
    }
}
