import Foundation

struct ResponsesRequest: Encodable, Sendable {
    let model: String?
    let instructions: String?
    let input: [ResponsesInputItem]
    let tools: [ResponsesToolDefinition]?
    let stream: Bool?
    let maxOutputTokens: Int?
    let text: ResponsesTextConfig?
    let store: Bool
    let reasoning: ResponsesReasoningConfig?
    let include: [String]?
    let previousResponseId: String?
    let extraFields: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case model, instructions, input, tools, stream, text, store, reasoning, include
        case maxOutputTokens = "max_output_tokens"
        case previousResponseId = "previous_response_id"
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(model, forKey: .model)
        try container.encodeIfPresent(instructions, forKey: .instructions)
        try container.encode(input, forKey: .input)
        try container.encodeIfPresent(tools, forKey: .tools)
        try container.encodeIfPresent(stream, forKey: .stream)
        try container.encodeIfPresent(maxOutputTokens, forKey: .maxOutputTokens)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encode(store, forKey: .store)
        try container.encodeIfPresent(reasoning, forKey: .reasoning)
        try container.encodeIfPresent(include, forKey: .include)
        try container.encodeIfPresent(previousResponseId, forKey: .previousResponseId)

        if !extraFields.isEmpty {
            var dynamicContainer = encoder.container(keyedBy: DynamicCodingKey.self)
            for (key, value) in extraFields {
                try dynamicContainer.encode(value, forKey: DynamicCodingKey(key))
            }
        }
    }
}

enum ResponsesInputItem: Encodable, Sendable {
    case userMessage(role: String, content: String)
    case assistantMessage(ResponsesAssistantItem)
    case functionCall(ResponsesFunctionCallItem)
    case functionCallOutput(ResponsesFunctionCallOutputItem)
    case reasoning(JSONValue)

    func encode(to encoder: any Encoder) throws {
        switch self {
        case let .userMessage(role, content):
            var container = encoder.container(keyedBy: UserMessageKeys.self)
            try container.encode("message", forKey: .type)
            try container.encode(role, forKey: .role)
            try container.encode(content, forKey: .content)
        case let .assistantMessage(item):
            try item.encode(to: encoder)
        case let .functionCall(item):
            try item.encode(to: encoder)
        case let .functionCallOutput(item):
            try item.encode(to: encoder)
        case let .reasoning(value):
            try value.encode(to: encoder)
        }
    }

    private enum UserMessageKeys: String, CodingKey {
        case type, role, content
    }
}

struct ResponsesAssistantItem: Encodable, Sendable {
    let type = "message"
    let role = "assistant"
    let content: [ResponsesOutputTextItem]

    enum CodingKeys: String, CodingKey {
        case type, role, content
    }
}

struct ResponsesOutputTextItem: Encodable, Sendable {
    let type = "output_text"
    let text: String

    enum CodingKeys: String, CodingKey {
        case type, text
    }
}

struct ResponsesFunctionCallItem: Encodable, Sendable {
    let type = "function_call"
    let callId: String
    let name: String
    let arguments: String

    enum CodingKeys: String, CodingKey {
        case type
        case callId = "call_id"
        case name, arguments
    }
}

struct ResponsesFunctionCallOutputItem: Encodable, Sendable {
    let type = "function_call_output"
    let callId: String
    let output: String

    enum CodingKeys: String, CodingKey {
        case type
        case callId = "call_id"
        case output
    }
}

struct ResponsesToolDefinition: Encodable, Sendable {
    let type = "function"
    let name: String
    let description: String
    let parameters: JSONSchema

    enum CodingKeys: String, CodingKey {
        case type, name, description, parameters
    }

    init(_ definition: ToolDefinition) {
        name = definition.name
        description = definition.description
        parameters = definition.parametersSchema
    }
}

struct ResponsesTextConfig: Encodable, Sendable {
    let format: ResponsesFormatConfig
}

struct ResponsesFormatConfig: Encodable, Sendable {
    let type = "json_schema"
    let name: String
    let strict = true
    let schema: JSONSchema

    enum CodingKeys: String, CodingKey {
        case type, name, strict, schema
    }
}

struct ResponsesReasoningConfig: Encodable, Sendable {
    let effort: String
    let summary: String?

    enum CodingKeys: String, CodingKey {
        case effort, summary
    }

    init(_ config: ReasoningConfig) {
        effort = config.effort.rawValue
        summary = config.exclude == true ? "disabled" : "auto"
    }
}

struct ResponsesAPIResponse: Decodable, Sendable {
    let id: String
    let status: String?
    let output: [ResponsesOutputItem]
    let usage: ResponsesUsage?
    let error: ResponsesErrorDetail?
}

enum ResponsesOutputItem: Decodable, Sendable {
    case message(ResponsesMessageOutput)
    case functionCall(ResponsesFunctionCallOutput)
    case reasoning(JSONValue)

    private enum TypeKey: String, CodingKey {
        case type
    }

    init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: TypeKey.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "message":
            self = try .message(ResponsesMessageOutput(from: decoder))
        case "function_call":
            self = try .functionCall(ResponsesFunctionCallOutput(from: decoder))
        default:
            self = try .reasoning(JSONValue(from: decoder))
        }
    }
}

struct ResponsesMessageOutput: Decodable, Sendable {
    let content: [ResponsesOutputContent]
}

struct ResponsesOutputContent: Decodable, Sendable {
    let type: String
    let text: String?
}

struct ResponsesFunctionCallOutput: Decodable, Sendable {
    let callId: String
    let name: String
    let arguments: String

    enum CodingKeys: String, CodingKey {
        case callId = "call_id"
        case name, arguments
    }
}

struct ResponsesUsage: Decodable, Sendable {
    let inputTokens: Int
    let outputTokens: Int
    let outputTokensDetails: ResponsesOutputTokensDetails?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case outputTokensDetails = "output_tokens_details"
    }
}

struct ResponsesOutputTokensDetails: Decodable, Sendable {
    let reasoningTokens: Int?

    enum CodingKeys: String, CodingKey {
        case reasoningTokens = "reasoning_tokens"
    }
}

struct ResponsesErrorDetail: Decodable, Sendable {
    let message: String
    let code: String
}
