@testable import AgentRunKit
import Foundation
import Testing

private let vertexProjectID = ProcessInfo.processInfo.environment["VERTEX_PROJECT_ID"] ?? ""
private let vertexLocation = ProcessInfo.processInfo.environment["VERTEX_LOCATION"] ?? ""
private let googleModel = ProcessInfo.processInfo.environment["SMOKE_VERTEX_GOOGLE_MODEL"] ?? ""
private let hasVertexGoogleConfig =
    !vertexProjectID.isEmpty
        && !vertexLocation.isEmpty
        && !googleModel.isEmpty
        && GoogleAuthService.credentialsAvailable()

@Suite(.enabled(
    if: hasVertexGoogleConfig,
    "Requires VERTEX_PROJECT_ID, VERTEX_LOCATION, SMOKE_VERTEX_GOOGLE_MODEL, and Google ADC credentials"
))
struct VertexGoogleSmokeTests {
    private func makeClient() throws -> VertexGoogleClient {
        try VertexGoogleClient(
            projectID: vertexProjectID,
            location: vertexLocation,
            model: googleModel,
            authService: GoogleAuthService(),
            maxOutputTokens: 1024,
            contextWindowSize: 100
        )
    }

    private func run<Client: LLMClient>(
        test testName: String = #function,
        using client: Client,
        _ body: (Client) async throws -> Void
    ) async throws {
        try await runSmoke(
            target: "vertex_google",
            test: testName,
            provider: "vertex",
            model: googleModel,
            using: client,
            body
        )
    }

    @Test
    func budgetHistoryIntegrity() async throws {
        try await run(using: makeClient()) { client in
            try await assertSmokeBudgetHistoryIntegrity(client: client)
        }
    }
}
