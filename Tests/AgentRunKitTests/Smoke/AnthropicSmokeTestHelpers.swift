@testable import AgentRunKit
import Foundation
import Testing

func assertSmokeAnthropicContinuityReplay(client: some LLMClient) async throws {
    let addTool = try makeSmokeAddTool()
    let configuration = AgentConfiguration(
        maxIterations: 5,
        systemPrompt: """
        You are a calculator assistant. When asked to add numbers, call the add tool exactly once.
        After receiving the tool result, immediately call finish with the numeric answer.
        Do not answer in plain text and do not call add again.
        """
    )
    let agent = Agent<EmptyContext>(client: client, tools: [addTool], configuration: configuration)

    var firstHistory: [ChatMessage]?
    for try await event in agent.stream(userMessage: "What is 17 + 25?", context: EmptyContext()) {
        guard case let .finished(_, _, _, history) = event.kind else { continue }
        firstHistory = history
    }
    let history = try smokeRequire(firstHistory)

    let replayableAssistant = history.compactMap { message -> AssistantMessage? in
        guard case let .assistant(assistant) = message else { return nil }
        guard assistant.continuity?.substrate == .anthropicMessages else { return nil }
        return assistant
    }.first
    let continuity = try smokeRequire(replayableAssistant?.continuity)

    try smokeExpect(continuity.substrate == .anthropicMessages)
    guard case let .object(payload) = continuity.payload,
          case let .array(blocks) = payload["content"]
    else {
        try smokeFail("Expected Anthropic continuity payload with content array")
    }
    try smokeExpect(!blocks.isEmpty)

    let secondResult = try await agent.run(
        userMessage: "What was the previous sum? Reply with just the number.",
        history: history,
        context: EmptyContext()
    )
    try smokeExpect(requireContent(secondResult).contains("42"))
}
