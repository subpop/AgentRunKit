@testable import AgentRunKit
import Foundation
import Testing

func assertSmokeResponsesContinuityReplay(client: ResponsesAPIClient) async throws {
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
    let firstHistory = try await streamSmokeAgentHistory(
        agent: agent,
        userMessage: "What is 17 + 25?",
        context: EmptyContext()
    )

    let replayableAssistant = try #require(firstHistory.compactMap { message -> AssistantMessage? in
        guard case let .assistant(assistant) = message else { return nil }
        guard assistant.toolCalls.contains(where: { $0.name == "add" }) else { return nil }
        return assistant
    }.last)
    let continuity = try #require(replayableAssistant.continuity)

    #expect(continuity.substrate == .responses)
    guard case let .object(payload) = continuity.payload,
          case let .array(output) = payload["output"]
    else {
        Issue.record("Expected Responses continuity payload")
        return
    }
    #expect(!output.isEmpty)

    let secondResult = try await agent.run(
        userMessage: "What was the previous sum? Reply with just the number.",
        history: firstHistory,
        context: EmptyContext()
    )
    #expect(try requireContent(secondResult).contains("42"))
}

func assertSmokeResponsesFinishSanitization(client: ResponsesAPIClient) async throws {
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
    let firstHistory = try await streamSmokeAgentHistory(
        agent: agent,
        userMessage: "What is 17 + 25?",
        context: EmptyContext()
    )

    try firstHistory.validateForAgentHistory()
    let assistantMessages = firstHistory.compactMap { message -> AssistantMessage? in
        guard case let .assistant(assistant) = message else { return nil }
        return assistant
    }
    #expect(!assistantMessages.isEmpty)
    #expect(assistantMessages.allSatisfy { assistant in
        !assistant.toolCalls.contains(where: { $0.name == "finish" })
    })

    let secondResult = try await agent.run(
        userMessage: "What was the previous sum? Reply with just the number.",
        history: firstHistory,
        context: EmptyContext()
    )

    #expect(try requireContent(secondResult).contains("42"))
}

private func streamSmokeAgentHistory<C: ToolContext>(
    agent: Agent<C>,
    userMessage: String,
    context: C
) async throws -> [ChatMessage] {
    var finishedHistory: [ChatMessage]?
    for try await event in agent.stream(userMessage: userMessage, context: context) {
        guard case let .finished(_, _, _, history) = event.kind else { continue }
        finishedHistory = history
    }
    return try #require(finishedHistory)
}
