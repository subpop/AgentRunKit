@testable import AgentRunKit
import Foundation

func sendSmokeStructuredOutput<T: Decodable & SchemaProviding>(
    _ prompt: String,
    client: any LLMClient,
    returning type: T.Type
) async throws -> T {
    let chat = Chat<EmptyContext>(client: client)
    do {
        let (result, _) = try await chat.send(prompt, returning: type)
        return result
    } catch let error as AgentError {
        guard case .structuredOutputDecodingFailed = error else {
            throw error
        }
        let response = try await client.generate(
            messages: [.user(prompt)],
            tools: [],
            responseFormat: .jsonSchema(T.self)
        )
        throw SmokeStructuredOutputFailure(
            rawContent: response.content,
            underlyingDescription: String(describing: error)
        )
    }
}
