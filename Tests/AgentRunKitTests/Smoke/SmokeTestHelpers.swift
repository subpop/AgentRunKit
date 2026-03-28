@testable import AgentRunKit
import Foundation
import Testing

struct SmokeAddParams: Codable, SchemaProviding {
    let lhs: Int
    let rhs: Int

    static var jsonSchema: JSONSchema {
        .object(
            properties: [
                "lhs": .integer(description: "First number"),
                "rhs": .integer(description: "Second number"),
            ],
            required: ["lhs", "rhs"]
        )
    }
}

struct SmokeAddOutput: Codable {
    let sum: Int
}

struct SmokeWorkout: Codable, SchemaProviding {
    let exercises: [SmokeExercise]

    static var jsonSchema: JSONSchema {
        .object(
            properties: [
                "exercises": .array(items: SmokeExercise.jsonSchema),
            ],
            required: ["exercises"]
        )
    }
}

struct SmokeExercise: Codable, SchemaProviding {
    let name: String
    let sets: Int

    static var jsonSchema: JSONSchema {
        .object(
            properties: [
                "name": .string(description: "Exercise name"),
                "sets": .integer(description: "Number of sets"),
            ],
            required: ["name", "sets"]
        )
    }
}

var smokeWeatherTool: ToolDefinition {
    ToolDefinition(
        name: "get_weather",
        description: "Get the current weather for a city",
        parametersSchema: .object(
            properties: ["city": .string(description: "The city name")],
            required: ["city"]
        )
    )
}

func makeSmokeAddTool() throws -> Tool<SmokeAddParams, SmokeAddOutput, EmptyContext> {
    try Tool(
        name: "add",
        description: "Add two numbers together. Always use this tool for addition.",
        executor: { params, _ in SmokeAddOutput(sum: params.lhs + params.rhs) }
    )
}

// MARK: - Assertions

func assertSmokeGenerate(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Be concise."),
        .user("What is 2 + 2? Reply with just the number."),
    ]
    let response = try await client.generate(messages: messages, tools: [])
    #expect(!response.content.isEmpty)
    #expect(response.content.contains("4"))
    #expect(response.toolCalls.isEmpty)
}

func assertSmokeStream(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Be concise."),
        .user("Count from 1 to 5, one number per line."),
    ]

    var contentChunks: [String] = []
    var finishedCount = 0

    for try await delta in client.stream(messages: messages, tools: []) {
        switch delta {
        case let .content(text):
            contentChunks.append(text)
        case .finished:
            finishedCount += 1
        default:
            break
        }
    }

    let fullContent = contentChunks.joined()
    #expect(!contentChunks.isEmpty)
    #expect(fullContent.contains("1"))
    #expect(fullContent.contains("5"))
    #expect(finishedCount >= 1)
}

func assertSmokeToolCall(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Use tools when appropriate."),
        .user("What's the weather in Tokyo?"),
    ]

    let response = try await client.generate(messages: messages, tools: [smokeWeatherTool])

    #expect(response.toolCalls.count >= 1)
    let toolCall = response.toolCalls.first { $0.name == "get_weather" }
    #expect(toolCall != nil)
    #expect(toolCall?.arguments.lowercased().contains("tokyo") == true)
}

func assertSmokeStreamingToolCall(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Use tools when appropriate."),
        .user("What's the weather in Paris?"),
    ]

    var toolCallStartName: String?
    var toolCallArgs: [Int: String] = [:]

    for try await delta in client.stream(messages: messages, tools: [smokeWeatherTool]) {
        switch delta {
        case let .toolCallStart(index, _, name):
            if name == "get_weather" {
                toolCallStartName = name
                toolCallArgs[index] = ""
            }
        case let .toolCallDelta(index, arguments):
            toolCallArgs[index, default: ""] += arguments
        default:
            break
        }
    }

    #expect(toolCallStartName == "get_weather")
    let allArgs = toolCallArgs.values.joined()
    #expect(allArgs.lowercased().contains("paris"))
}

func assertSmokeTokenUsage(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Be concise."),
        .user("What is 2 + 2? Reply with just the number."),
    ]
    let response = try await client.generate(messages: messages, tools: [])
    #expect(response.tokenUsage != nil)
    #expect(response.tokenUsage?.input ?? 0 > 0)
    #expect(response.tokenUsage?.output ?? 0 > 0)
}

func assertSmokeStreamingTokenUsage(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Be concise."),
        .user("What is 2 + 2? Reply with just the number."),
    ]

    var streamUsage: TokenUsage?

    for try await delta in client.stream(messages: messages, tools: []) {
        if case let .finished(usage) = delta {
            streamUsage = usage
        }
    }

    #expect(streamUsage != nil)
    #expect(streamUsage?.input ?? 0 > 0)
    #expect(streamUsage?.output ?? 0 > 0)
}

func assertSmokeStructuredOutput(client: any LLMClient) async throws {
    let chat = Chat<EmptyContext>(client: client)
    let (workout, history) = try await chat.send(
        "Give me exactly 2 upper body exercises",
        returning: SmokeWorkout.self
    )

    #expect(workout.exercises.count >= 2)
    for exercise in workout.exercises {
        #expect(!exercise.name.isEmpty)
        #expect(exercise.sets > 0)
    }
    #expect(history.count >= 2)
}

func assertSmokeAgentLoop(client: any LLMClient) async throws {
    let addTool = try makeSmokeAddTool()
    let config = AgentConfiguration(
        maxIterations: 5,
        systemPrompt: """
        You are a calculator assistant. When asked to add numbers, use the add tool.
        After getting the result, use the finish tool with the answer.
        """
    )

    let agent = Agent<EmptyContext>(client: client, tools: [addTool], configuration: config)
    let result = try await agent.run(userMessage: "What is 17 + 25?", context: EmptyContext())

    #expect(result.content.contains("42"))
    #expect(result.iterations >= 1)
}

func assertSmokeStreamingAgentLoop(client: any LLMClient) async throws {
    let addTool = try makeSmokeAddTool()
    let config = AgentConfiguration(
        maxIterations: 5,
        systemPrompt: """
        You are a calculator assistant. When asked to add numbers, use the add tool.
        After getting the result, use the finish tool with the answer.
        """
    )

    let agent = Agent<EmptyContext>(client: client, tools: [addTool], configuration: config)

    var toolStarted = false
    var toolCompleted = false
    var hasFinished = false
    var finishContent: String?

    for try await event in agent.stream(userMessage: "What is 7 + 8?", context: EmptyContext()) {
        switch event {
        case .toolCallStarted:
            toolStarted = true
        case let .toolCallCompleted(_, name, result):
            if name == "add" {
                toolCompleted = true
                #expect(result.content.contains("15"))
            }
        case let .finished(_, content, _, _):
            hasFinished = true
            finishContent = content
        default:
            break
        }
    }

    #expect(toolStarted)
    #expect(toolCompleted)
    #expect(hasFinished)
    #expect(finishContent?.contains("15") == true)
}

func assertSmokeMultiTurn(client: any LLMClient) async throws {
    var messages: [ChatMessage] = [
        .system("You are a helpful assistant. Be concise."),
        .user("Remember the number 42."),
    ]

    let response1 = try await client.generate(messages: messages, tools: [])
    messages.append(.assistant(response1))
    messages.append(.user("What number did I ask you to remember?"))

    let response2 = try await client.generate(messages: messages, tools: [])
    #expect(response2.content.contains("42"))
}

func assertSmokeReasoningGenerate(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant."),
        .user("What is 5 + 7?"),
    ]

    let response = try await client.generate(messages: messages, tools: [])
    #expect(response.content.contains("12"))
    #expect(response.tokenUsage != nil)
    #expect(response.reasoning != nil)
}

func assertSmokeReasoningStream(client: any LLMClient) async throws {
    let messages: [ChatMessage] = [
        .system("You are a helpful assistant. Think step by step."),
        .user("What is 17 * 23? Show your reasoning."),
    ]

    var contentChunks: [String] = []
    var reasoningChunks: [String] = []
    var hasFinished = false

    for try await delta in client.stream(messages: messages, tools: []) {
        switch delta {
        case let .content(text):
            contentChunks.append(text)
        case let .reasoning(text):
            reasoningChunks.append(text)
        case .finished:
            hasFinished = true
        default:
            break
        }
    }

    let fullContent = contentChunks.joined()
    #expect(hasFinished)
    #expect(fullContent.contains("391"))
    #expect(!reasoningChunks.isEmpty)
}
