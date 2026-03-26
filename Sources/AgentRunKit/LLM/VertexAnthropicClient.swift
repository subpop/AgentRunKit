import Foundation

/// An LLM client for Anthropic Claude models served via Vertex AI.
///
/// Uses OAuth2 Bearer token authentication (via ``GoogleAuthService`` or a custom
/// token provider closure) instead of Anthropic API key authentication.
///
/// The wire format is the standard Anthropic Messages API with a
/// `"anthropic_version": "vertex-2023-10-16"` field injected into the request body.
/// Response parsing and SSE streaming are delegated to an internal ``AnthropicClient``.
///
/// ```swift
/// let auth = try GoogleAuthService()
/// let client = VertexAnthropicClient(
///     projectID: "my-project",
///     location: "us-east5",
///     model: "claude-sonnet-4-6",
///     authService: auth
/// )
/// ```
public struct VertexAnthropicClient: LLMClient, Sendable {
    public let contextWindowSize: Int?

    let anthropic: AnthropicClient
    private let projectID: String
    private let location: String
    private let model: String
    private let tokenProvider: @Sendable () async throws -> String
    private let session: URLSession
    private let retryPolicy: RetryPolicy

    public init(
        projectID: String,
        location: String,
        model: String,
        tokenProvider: @Sendable @escaping () async throws -> String,
        maxTokens: Int = 8192,
        contextWindowSize: Int? = nil,
        session: URLSession = .shared,
        retryPolicy: RetryPolicy = .default,
        reasoningConfig: ReasoningConfig? = nil,
        interleavedThinking: Bool = true,
        cachingEnabled: Bool = false
    ) {
        self.projectID = projectID
        self.location = location
        self.model = model
        self.tokenProvider = tokenProvider
        self.session = session
        self.retryPolicy = retryPolicy
        self.contextWindowSize = contextWindowSize
        anthropic = AnthropicClient(
            apiKey: "",
            model: model,
            maxTokens: maxTokens,
            contextWindowSize: contextWindowSize,
            session: session,
            retryPolicy: retryPolicy,
            reasoningConfig: reasoningConfig,
            interleavedThinking: interleavedThinking,
            cachingEnabled: cachingEnabled
        )
    }

    /// Convenience initializer that uses a ``GoogleAuthService`` for authentication.
    public init(
        projectID: String,
        location: String,
        model: String,
        authService: GoogleAuthService,
        maxTokens: Int = 8192,
        contextWindowSize: Int? = nil,
        session: URLSession = .shared,
        retryPolicy: RetryPolicy = .default,
        reasoningConfig: ReasoningConfig? = nil,
        interleavedThinking: Bool = true,
        cachingEnabled: Bool = false
    ) {
        self.init(
            projectID: projectID,
            location: location,
            model: model,
            tokenProvider: { try await authService.accessToken() },
            maxTokens: maxTokens,
            contextWindowSize: contextWindowSize,
            session: session,
            retryPolicy: retryPolicy,
            reasoningConfig: reasoningConfig,
            interleavedThinking: interleavedThinking,
            cachingEnabled: cachingEnabled
        )
    }

    // MARK: - LLMClient

    public func generate(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        responseFormat: ResponseFormat?,
        requestContext: RequestContext?
    ) async throws -> AssistantMessage {
        if responseFormat != nil {
            throw AgentError.llmError(.other("VertexAnthropicClient does not support responseFormat"))
        }
        let request = try anthropic.buildRequest(
            messages: messages,
            tools: tools,
            extraFields: requestContext?.extraFields ?? [:]
        )
        let token = try await tokenProvider()
        let urlRequest = try buildVertexURLRequest(
            VertexAnthropicRequest(inner: request), stream: false, token: token
        )
        let (data, httpResponse) = try await HTTPRetry.performData(
            urlRequest: urlRequest, session: session, retryPolicy: retryPolicy
        )
        requestContext?.onResponse?(httpResponse)
        return try anthropic.parseResponse(data)
    }

    public func stream(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        requestContext: RequestContext?
    ) -> AsyncThrowingStream<StreamDelta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await performStreamRequest(
                        messages: messages,
                        tools: tools,
                        extraFields: requestContext?.extraFields ?? [:],
                        onResponse: requestContext?.onResponse,
                        continuation: continuation
                    )
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Streaming

    private func performStreamRequest(
        messages: [ChatMessage],
        tools: [ToolDefinition],
        extraFields: [String: JSONValue],
        onResponse: (@Sendable (HTTPURLResponse) -> Void)?,
        continuation: AsyncThrowingStream<StreamDelta, Error>.Continuation
    ) async throws {
        let request = try anthropic.buildRequest(
            messages: messages, tools: tools,
            stream: true, extraFields: extraFields
        )
        let token = try await tokenProvider()
        let urlRequest = try buildVertexURLRequest(
            VertexAnthropicRequest(inner: request), stream: true, token: token
        )
        let (bytes, httpResponse) = try await HTTPRetry.performStream(
            urlRequest: urlRequest, session: session, retryPolicy: retryPolicy
        )
        onResponse?(httpResponse)

        let state = AnthropicStreamState()

        try await processSSEStream(
            bytes: bytes,
            stallTimeout: retryPolicy.streamStallTimeout
        ) { line in
            try await anthropic.handleSSELine(
                line, state: state, continuation: continuation
            )
        }
        continuation.finish()
    }

    // MARK: - URL Construction

    func buildVertexURLRequest(
        _ request: VertexAnthropicRequest,
        stream: Bool,
        token: String
    ) throws -> URLRequest {
        let action = stream ? "streamRawPredict" : "rawPredict"
        let basePath = "v1/projects/\(projectID)/locations/\(location)"
            + "/publishers/anthropic/models/\(model):\(action)"
        let baseURL = URL(string: "https://\(location)-aiplatform.googleapis.com")!
        let url = baseURL.appendingPathComponent(basePath)

        let headers = ["Authorization": "Bearer \(token)"]
        return try buildJSONPostRequest(url: url, body: request, headers: headers)
    }
}

// MARK: - Vertex Anthropic Request Wrapper

/// Wraps an ``AnthropicRequest`` and injects `"anthropic_version": "vertex-2023-10-16"`
/// into the encoded JSON body for Vertex AI compatibility.
struct VertexAnthropicRequest: Encodable {
    static let vertexAnthropicVersion = "vertex-2023-10-16"

    let inner: AnthropicRequest

    func encode(to encoder: any Encoder) throws {
        try inner.encode(to: encoder)
        var container = encoder.container(keyedBy: DynamicCodingKey.self)
        try container.encode(
            Self.vertexAnthropicVersion,
            forKey: DynamicCodingKey("anthropic_version")
        )
    }
}
