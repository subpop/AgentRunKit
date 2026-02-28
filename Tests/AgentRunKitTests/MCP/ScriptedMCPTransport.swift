@testable import AgentRunKit
import Foundation

actor ScriptedMCPTransport: MCPTransport {
    private var responses: [Data]
    private var responseIndex = 0
    private let stream: AsyncThrowingStream<Data, Error>
    private let continuation: AsyncThrowingStream<Data, Error>.Continuation
    private var connected = false

    init(responses: [Data]) {
        self.responses = responses
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        self.stream = stream
        self.continuation = continuation
    }

    func connect() async throws { connected = true }

    func disconnect() async {
        connected = false
        continuation.finish()
    }

    func send(_ data: Data) async throws {
        guard connected else { throw MCPError.transportClosed }
        // Notifications (no "id" field) don't expect responses
        if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any], dict["id"] == nil {
            return
        }
        guard responseIndex < responses.count else {
            throw MCPError.connectionFailed("No more scripted responses (sent: \(responseIndex))")
        }
        let response = responses[responseIndex]
        responseIndex += 1
        continuation.yield(response)
    }

    nonisolated func messages() -> AsyncThrowingStream<Data, Error> { stream }
}
