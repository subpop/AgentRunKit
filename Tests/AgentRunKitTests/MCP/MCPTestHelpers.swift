@testable import AgentRunKit
import Foundation

// swiftlint:disable force_try
enum MCPTestHelpers {
    struct MockTool: Sendable {
        let name: String
        let description: String
        let schema: JSONValue
    }

    static func encodeResponse(id: Int, result: JSONValue) -> Data {
        let response = JSONRPCResponse(
            jsonrpc: "2.0",
            id: .int(id),
            result: result,
            error: nil
        )
        return try! JSONEncoder().encode(response)
    }

    static func encodeErrorResponse(id: Int, code: Int, message: String) -> Data {
        let response = JSONRPCResponse(
            jsonrpc: "2.0",
            id: .int(id),
            result: nil,
            error: JSONRPCErrorObject(code: code, message: message, data: nil)
        )
        return try! JSONEncoder().encode(response)
    }

    static func encodeNotification(method: String, params: JSONValue? = nil) -> Data {
        let notification = JSONRPCNotification(method: method, params: params)
        return try! JSONEncoder().encode(notification)
    }

    static func initializeResult(protocolVersion: String = "2025-06-18") -> JSONValue {
        .object([
            "protocolVersion": .string(protocolVersion),
            "capabilities": .object([:]),
            "serverInfo": .object([
                "name": .string("test-server"),
                "version": .string("1.20.1"),
            ]),
        ])
    }

    static func toolsListResult(
        tools: [MockTool],
        nextCursor: String? = nil
    ) -> JSONValue {
        var dict: [String: JSONValue] = [
            "tools": .array(tools.map { tool in
                .object([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "inputSchema": tool.schema,
                ])
            }),
        ]
        if let nextCursor {
            dict["nextCursor"] = .string(nextCursor)
        }
        return .object(dict)
    }

    static func callToolResult(text: String, isError: Bool = false) -> JSONValue {
        .object([
            "content": .array([
                .object([
                    "type": .string("text"),
                    "text": .string(text),
                ]),
            ]),
            "isError": .bool(isError),
        ])
    }

    static func emptyToolsListResult() -> JSONValue {
        .object(["tools": .array([])])
    }

    static func toolSchema(properties: [String: JSONValue], required: [String] = []) -> JSONValue {
        .object([
            "type": .string("object"),
            "properties": .object(properties),
            "required": .array(required.map { .string($0) }),
        ])
    }
}

// swiftlint:enable force_try
