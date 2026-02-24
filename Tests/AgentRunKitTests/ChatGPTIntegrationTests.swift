import Foundation
import Testing

@testable import AgentRunKit

private struct CodexAuth: Decodable {
    struct Tokens: Decodable {
        let accessToken: String
        let accountId: String

        enum CodingKeys: String, CodingKey {
            case accessToken = "access_token"
            case accountId = "account_id"
        }
    }

    let tokens: Tokens
}

private let codexAuth: CodexAuth? = {
    let path = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".codex/auth.json")
    guard let data = try? Data(contentsOf: path) else { return nil }
    return try? JSONDecoder().decode(CodexAuth.self, from: data)
}()

private let hasAuth = codexAuth != nil

private struct AddParams: Codable, SchemaProviding, Sendable {
    let lhs: Int
    let rhs: Int

    static var jsonSchema: JSONSchema {
        .object(
            properties: [
                "lhs": .integer(description: "First number"),
                "rhs": .integer(description: "Second number")
            ],
            required: ["lhs", "rhs"]
        )
    }
}

private struct AddOutput: Codable, Sendable {
    let sum: Int
}

@Suite(.enabled(if: hasAuth, "Requires ~/.codex/auth.json (run Codex CLI to authenticate)"))
struct ChatGPTIntegrationTests {
    private func makeClient() -> ResponsesAPIClient {
        let auth = codexAuth!
        return ResponsesAPIClient(
            model: "gpt-5.2",
            maxOutputTokens: nil,
            baseURL: ResponsesAPIClient.chatGPTBaseURL,
            additionalHeaders: {
                [
                    "Authorization": "Bearer \(auth.tokens.accessToken)",
                    "ChatGPT-Account-ID": auth.tokens.accountId
                ]
            },
            store: false
        )
    }

    @Test
    func streamingCompletion() async throws {
        let agent = Agent<EmptyContext>(
            client: makeClient(),
            tools: [],
            configuration: AgentConfiguration(
                maxIterations: 3,
                systemPrompt: "You are a helpful assistant. Be concise."
            )
        )

        var events: [StreamEvent] = []
        for try await event in agent.stream(
            userMessage: "Say hello and finish. Use the finish tool with content 'Hello!'",
            context: EmptyContext()
        ) {
            events.append(event)
        }

        let finished = events.compactMap { event -> String? in
            if case let .finished(_, content, _, _) = event { return content }
            return nil
        }.first
        #expect(finished != nil, "Expected a .finished event")
        #expect(finished?.lowercased().contains("hello") == true)
    }

    @Test
    func streamingToolCallWithReasoning() async throws {
        let addTool = try Tool<AddParams, AddOutput, EmptyContext>(
            name: "add",
            description: "Add two numbers together. Always use this tool for addition.",
            executor: { params, _ in AddOutput(sum: params.lhs + params.rhs) }
        )

        let agent = Agent<EmptyContext>(
            client: makeClient(),
            tools: [addTool],
            configuration: AgentConfiguration(
                maxIterations: 5,
                systemPrompt: """
                You are a calculator assistant. When asked to add numbers, use the add tool.
                After getting the result, use the finish tool with the answer.
                """
            )
        )

        var events: [StreamEvent] = []
        for try await event in agent.stream(
            userMessage: "What is 17 + 25?",
            context: EmptyContext()
        ) {
            events.append(event)
        }

        let toolCompleted = events.contains { event in
            if case .toolCallCompleted = event { return true }
            return false
        }
        #expect(toolCompleted, "Expected at least one .toolCallCompleted event")

        let finished = events.compactMap { event -> (content: String?, usage: TokenUsage)? in
            if case let .finished(usage, content, _, _) = event {
                return (content, usage)
            }
            return nil
        }.first
        #expect(finished != nil, "Expected a .finished event")
        #expect(finished?.content?.contains("42") == true)
        #expect((finished?.usage.input ?? 0) > 0)
        #expect((finished?.usage.output ?? 0) > 0)
    }
}
