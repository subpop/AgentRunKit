import Foundation

extension OpenAIClient {
    func performStreamRequest(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        extraFields: [String: JSONValue],
        onResponse: (@Sendable (HTTPURLResponse) -> Void)?,
        continuation: AsyncThrowingStream<StreamDelta, Error>.Continuation
    ) async throws {
        let request = buildRequest(messages: messages, tools: tools, stream: true, extraFields: extraFields)
        let urlRequest = try buildURLRequest(request)
        let (bytes, httpResponse) = try await HTTPRetry.performStream(
            urlRequest: urlRequest, session: session, retryPolicy: retryPolicy
        )
        onResponse?(httpResponse)
        if let stallTimeout = retryPolicy.streamStallTimeout {
            try await processStreamWithStallDetection(
                bytes: bytes, stallTimeout: stallTimeout, continuation: continuation
            )
        } else {
            try await processStreamLines(bytes: bytes, continuation: continuation)
        }
    }

    private func handleSSELine(
        _ line: String,
        continuation: AsyncThrowingStream<StreamDelta, Error>.Continuation
    ) throws -> Bool {
        try Task.checkCancellation()
        guard let payload = Self.extractSSEPayload(from: line) else { return false }
        if payload == "[DONE]" {
            continuation.finish()
            return true
        }
        let chunkData = Data(payload.utf8)
        let chunk = try parseStreamingChunk(chunkData)
        if let details = try JSONValue.extractReasoningDetails(from: chunkData) {
            continuation.yield(.reasoningDetails(details))
        }
        for delta in try extractDeltas(from: chunk) {
            continuation.yield(delta)
        }
        return false
    }

    func processStreamLines<S: AsyncSequence>(
        bytes: S,
        continuation: AsyncThrowingStream<StreamDelta, Error>.Continuation
    ) async throws where S.Element == UInt8 {
        for try await line in UnboundedLines(source: bytes) {
            if try handleSSELine(line, continuation: continuation) { return }
        }
        continuation.finish()
    }

    func processStreamWithStallDetection<S: AsyncSequence & Sendable>(
        bytes: S,
        stallTimeout: Duration,
        continuation: AsyncThrowingStream<StreamDelta, Error>.Continuation
    ) async throws where S.Element == UInt8 {
        try await withThrowingTaskGroup(of: Void.self) { group in
            let watchdog = StallWatchdog()

            group.addTask {
                while !Task.isCancelled {
                    let snapshot = await watchdog.lastActivity
                    try await Task.sleep(for: stallTimeout)
                    let current = await watchdog.lastActivity
                    if current == snapshot {
                        throw AgentError.llmError(.streamStalled)
                    }
                }
            }

            group.addTask { [self] in
                for try await line in UnboundedLines(source: bytes) {
                    await watchdog.recordActivity()
                    if try handleSSELine(line, continuation: continuation) { return }
                }
                continuation.finish()
            }

            try await group.next()
            group.cancelAll()
        }
    }

    func performUploadWithRetry<T>(
        urlRequest: URLRequest,
        bodyFileURL: URL,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)? = nil,
        onSuccess: (Data, HTTPURLResponse) throws -> T
    ) async throws -> T {
        var lastError: (any Error)?
        var sleptForRetryAfter = false

        for attempt in 0 ..< retryPolicy.maxAttempts {
            try Task.checkCancellation()
            if attempt > 0, !sleptForRetryAfter {
                try await Task.sleep(for: retryPolicy.delay(forAttempt: attempt - 1))
            }
            sleptForRetryAfter = false

            let data: Data
            let response: URLResponse
            do {
                (data, response) = try await session.upload(for: urlRequest, fromFile: bodyFileURL)
            } catch {
                lastError = TransportError.networkError(error)
                continue
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw AgentError.llmError(.invalidResponse)
            }

            if (200 ... 299).contains(httpResponse.statusCode) {
                onResponse?(httpResponse)
                return try onSuccess(data, httpResponse)
            }

            let result = try await HTTPRetry.handleErrorStatus(
                httpResponse: httpResponse,
                errorBody: String(data: data, encoding: .utf8) ?? "",
                attempt: attempt,
                retryPolicy: retryPolicy,
                sleptForRetryAfter: &sleptForRetryAfter
            )

            switch result {
            case .continue: continue
            case let .stop(error): lastError = error
            }

            if !retryPolicy.isRetryable(statusCode: httpResponse.statusCode) { break }
        }

        let transportError = lastError as? TransportError
            ?? .other(lastError.map { String(describing: $0) } ?? "Unknown error")
        throw AgentError.llmError(transportError)
    }

    static func extractSSEPayload(from line: String) -> String? {
        guard line.hasPrefix("data:") else { return nil }
        return line.hasPrefix("data: ")
            ? String(line.dropFirst(6))
            : String(line.dropFirst(5))
    }

    func parseStreamingChunk(_ data: Data) throws -> StreamingChunk {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do {
            return try decoder.decode(StreamingChunk.self, from: data)
        } catch {
            throw AgentError.llmError(.decodingFailed(error))
        }
    }

    func extractDeltas(from chunk: StreamingChunk) throws -> [StreamDelta] {
        var deltas: [StreamDelta] = []
        for choice in chunk.choices ?? [] {
            if let reasoning = choice.delta.reasoning ?? choice.delta.reasoningContent, !reasoning.isEmpty {
                deltas.append(.reasoning(reasoning))
            }
            if let content = choice.delta.content, !content.isEmpty {
                deltas.append(.content(content))
            }
            if let toolCalls = choice.delta.toolCalls {
                for call in toolCalls {
                    if let id = call.id, !id.isEmpty, let name = call.function?.name, !name.isEmpty {
                        deltas.append(.toolCallStart(index: call.index, id: id, name: name))
                    }
                    if let args = call.function?.arguments, !args.isEmpty {
                        deltas.append(.toolCallDelta(index: call.index, arguments: args))
                    }
                }
            }
            if let audio = choice.delta.audio {
                if let id = audio.id, !id.isEmpty {
                    deltas.append(.audioStarted(id: id, expiresAt: audio.expiresAt ?? 0))
                }
                if let base64 = audio.data, !base64.isEmpty {
                    guard let decoded = Data(base64Encoded: base64) else {
                        throw AgentError.llmError(.decodingFailed(
                            description: "Invalid base64 in audio data"
                        ))
                    }
                    deltas.append(.audioData(decoded))
                }
                if let transcript = audio.transcript, !transcript.isEmpty {
                    deltas.append(.audioTranscript(transcript))
                }
            }
            if choice.finishReason != nil {
                deltas.append(.finished(usage: chunk.usage.map { usage in
                    let reasoning = usage.completionTokensDetails?.reasoningTokens ?? 0
                    let output = max(0, usage.completionTokens - reasoning)
                    return TokenUsage(input: usage.promptTokens, output: output, reasoning: reasoning)
                }))
            }
        }
        return deltas
    }
}

actor StallWatchdog {
    private(set) var lastActivity: ContinuousClock.Instant = .now

    func recordActivity() {
        lastActivity = .now
    }
}

struct UnboundedLines<Source: AsyncSequence>: AsyncSequence where Source.Element == UInt8 {
    typealias Element = String
    let source: Source

    func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(sourceIterator: source.makeAsyncIterator())
    }

    struct AsyncIterator: AsyncIteratorProtocol {
        var sourceIterator: Source.AsyncIterator
        var buffer = Data(capacity: 4096)

        mutating func next() async throws -> String? {
            while true {
                guard let byte = try await sourceIterator.next() else {
                    if buffer.isEmpty { return nil }
                    return try decodeAndClear()
                }
                if byte == 0x0A {
                    return try decodeAndClear()
                }
                buffer.append(byte)
            }
        }

        private mutating func decodeAndClear() throws -> String {
            defer { buffer.removeAll(keepingCapacity: true) }
            if buffer.last == 0x0D { buffer.removeLast() }
            guard let line = String(data: buffer, encoding: .utf8) else {
                throw AgentError.llmError(.decodingFailed(description: "Invalid UTF-8 in SSE stream"))
            }
            return line
        }
    }
}
