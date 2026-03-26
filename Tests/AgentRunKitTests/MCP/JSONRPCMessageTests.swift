@testable import AgentRunKit
import Foundation
import Testing

struct JSONRPCMessageTests {
    @Test
    func requestWithIntIdRoundTrip() throws {
        let request = JSONRPCRequest(id: .int(42), method: "test", params: .object(["key": .string("value")]))
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(JSONRPCRequest.self, from: data)
        #expect(decoded.jsonrpc == "2.0")
        #expect(decoded.id == .int(42))
        #expect(decoded.method == "test")
        guard case let .object(params) = decoded.params, case let .string(val) = params["key"] else {
            Issue.record("Expected object params with key")
            return
        }
        #expect(val == "value")
    }

    @Test
    func requestWithStringIdRoundTrip() throws {
        let request = JSONRPCRequest(id: .string("abc-123"), method: "ping")
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(JSONRPCRequest.self, from: data)
        #expect(decoded.id == .string("abc-123"))
        #expect(decoded.method == "ping")
        #expect(decoded.params == nil)
    }

    @Test
    func responseWithResultRoundTrip() throws {
        let response = JSONRPCResponse(
            jsonrpc: "2.0",
            id: .int(1),
            result: .object(["data": .string("hello")]),
            error: nil
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(JSONRPCResponse.self, from: data)
        #expect(decoded.id == .int(1))
        guard case let .object(result) = decoded.result, case let .string(val) = result["data"] else {
            Issue.record("Expected object result")
            return
        }
        #expect(val == "hello")
        #expect(decoded.error == nil)
    }

    @Test
    func responseWithErrorRoundTrip() throws {
        let response = JSONRPCResponse(
            jsonrpc: "2.0",
            id: .int(1),
            result: nil,
            error: JSONRPCErrorObject(code: -32600, message: "Invalid Request", data: nil)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(JSONRPCResponse.self, from: data)
        #expect(decoded.result == nil)
        #expect(decoded.error?.code == -32600)
        #expect(decoded.error?.message == "Invalid Request")
    }

    @Test
    func notificationRoundTrip() throws {
        let notification = JSONRPCNotification(method: "update", params: .array([.int(1), .int(2)]))
        let data = try JSONEncoder().encode(notification)
        let decoded = try JSONDecoder().decode(JSONRPCNotification.self, from: data)
        #expect(decoded.jsonrpc == "2.0")
        #expect(decoded.method == "update")
        guard case let .array(items) = decoded.params else {
            Issue.record("Expected array params")
            return
        }
        #expect(items == [.int(1), .int(2)])
    }

    @Test
    func discriminatedDecodingRequest() throws {
        let json = #"{"jsonrpc":"2.0","id":1,"method":"test","params":{}}"#
        let message = try JSONDecoder().decode(JSONRPCMessage.self, from: Data(json.utf8))
        guard case let .request(req) = message else {
            Issue.record("Expected request, got \(message)")
            return
        }
        #expect(req.method == "test")
        #expect(req.id == .int(1))
    }

    @Test
    func discriminatedDecodingNotification() throws {
        let json = #"{"jsonrpc":"2.0","method":"notify"}"#
        let message = try JSONDecoder().decode(JSONRPCMessage.self, from: Data(json.utf8))
        guard case let .notification(notif) = message else {
            Issue.record("Expected notification, got \(message)")
            return
        }
        #expect(notif.method == "notify")
    }

    @Test
    func discriminatedDecodingResponse() throws {
        let json = #"{"jsonrpc":"2.0","id":5,"result":"ok"}"#
        let message = try JSONDecoder().decode(JSONRPCMessage.self, from: Data(json.utf8))
        guard case let .response(resp) = message else {
            Issue.record("Expected response, got \(message)")
            return
        }
        #expect(resp.id == .int(5))
    }

    @Test
    func errorWithDataField() throws {
        let errorObj = JSONRPCErrorObject(
            code: -32000,
            message: "Server error",
            data: .object(["detail": .string("info")])
        )
        let response = JSONRPCResponse(jsonrpc: "2.0", id: .int(1), result: nil, error: errorObj)
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(JSONRPCResponse.self, from: data)
        guard case let .object(errorData) = decoded.error?.data else {
            Issue.record("Expected error data object")
            return
        }
        #expect(errorData["detail"] == .string("info"))
    }

    @Test
    func parseErrorWithNullId() throws {
        let json = #"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#
        let message = try JSONDecoder().decode(JSONRPCMessage.self, from: Data(json.utf8))
        guard case let .response(resp) = message else {
            Issue.record("Expected response")
            return
        }
        #expect(resp.id == .null)
        #expect(resp.error?.code == -32700)
    }

    @Test
    func idNullDoesNotCollideWithIntNegOne() {
        #expect(JSONRPCID.null != JSONRPCID.int(-1))
    }

    @Test
    func initializeRequestEncoding() throws {
        let request = JSONRPCRequest(
            id: .int(1),
            method: "initialize",
            params: .object([
                "protocolVersion": .string("2025-06-18"),
                "capabilities": .object([:]),
                "clientInfo": .object([
                    "name": .string("AgentRunKit"),
                    "version": .string("1.20.1"),
                ]),
            ])
        )
        let data = try JSONEncoder().encode(request)
        let decoded = try JSONDecoder().decode(JSONRPCRequest.self, from: data)
        #expect(decoded.method == "initialize")
        guard case let .object(params) = decoded.params,
              case let .string(version) = params["protocolVersion"]
        else {
            Issue.record("Expected protocolVersion in params")
            return
        }
        #expect(version == "2025-06-18")
    }

    @Test
    func toolsCallResponseDecoding() throws {
        let json = """
        {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [{"type": "text", "text": "hello"}],
                "isError": true
            }
        }
        """
        let response = try JSONDecoder().decode(JSONRPCResponse.self, from: Data(json.utf8))
        guard case let .object(result) = response.result,
              case let .bool(isError) = result["isError"]
        else {
            Issue.record("Expected result with isError")
            return
        }
        #expect(isError == true)
    }

    @Test
    func toolsCallWithStructuredContent() throws {
        let json = """
        {
            "jsonrpc": "2.0",
            "id": 4,
            "result": {
                "content": [{"type": "text", "text": "summary"}],
                "structuredContent": {"key": "value"}
            }
        }
        """
        let response = try JSONDecoder().decode(JSONRPCResponse.self, from: Data(json.utf8))
        guard case let .object(result) = response.result,
              case let .object(structuredContent) = result["structuredContent"]
        else {
            Issue.record("Expected structuredContent")
            return
        }
        #expect(structuredContent["key"] == .string("value"))
    }

    @Test
    func requestParamsNilOmitted() throws {
        let request = JSONRPCRequest(id: .int(1), method: "ping")
        let data = try JSONEncoder().encode(request)
        let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(dict?["params"] == nil)
    }
}
