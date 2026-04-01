#if canImport(FoundationModels)

    import AgentRunKit
    @testable import AgentRunKitFoundationModels
    import Foundation
    import Testing

    struct FoundationModelsClientTests {
        @Test func contextWindowSize() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            #expect(client.contextWindowSize == 4096)
        }

        @Test func responseFormatThrows() async {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            await #expect(throws: AgentError.self) {
                try await client.generate(
                    messages: [.user("test")],
                    tools: [],
                    responseFormat: ResponseFormat.jsonSchema(DummySchema.self),
                    requestContext: nil
                )
            }
        }

        @Test func synthesizeFinishStructure() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let message = FoundationModelsClient<EmptyContext>.synthesizeFinish(content: "Hello")
            #expect(message.toolCalls.count == 1)
            #expect(message.toolCalls[0].id == "fm_finish")
            #expect(message.toolCalls[0].name == "finish")
            #expect(message.content.isEmpty)
        }

        @Test func synthesizeFinishRoundTrip() throws {
            guard #available(macOS 26, iOS 26, *) else { return }
            let message = FoundationModelsClient<EmptyContext>.synthesizeFinish(content: "Test content")
            let data = Data(message.toolCalls[0].arguments.utf8)
            let decoded = try JSONDecoder().decode(FinishArguments.self, from: data)
            #expect(decoded.content == "Test content")
        }

        @Test func synthesizeFinishSpecialCharacters() throws {
            guard #available(macOS 26, iOS 26, *) else { return }
            let content = #"She said "hello" & <goodbye> \n tab:\t end"#
            let message = FoundationModelsClient<EmptyContext>.synthesizeFinish(content: content)
            let data = Data(message.toolCalls[0].arguments.utf8)
            let decoded = try JSONDecoder().decode(FinishArguments.self, from: data)
            #expect(decoded.content == content)
        }

        @Test func synthesizeFinishEmptyContent() throws {
            guard #available(macOS 26, iOS 26, *) else { return }
            let message = FoundationModelsClient<EmptyContext>.synthesizeFinish(content: "")
            let data = Data(message.toolCalls[0].arguments.utf8)
            let decoded = try JSONDecoder().decode(FinishArguments.self, from: data)
            #expect(decoded.content.isEmpty)
        }

        @Test func mergeInstructionsBothPresent() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(
                context: EmptyContext(), instructions: "Base"
            )
            let result = client.mergeInstructions("FromMessages")
            #expect(result == "Base\nFromMessages")
        }

        @Test func mergeInstructionsBaseOnly() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(
                context: EmptyContext(), instructions: "Base"
            )
            #expect(client.mergeInstructions(nil) == "Base")
        }

        @Test func mergeInstructionsMessageOnly() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            #expect(client.mergeInstructions("FromMessages") == "FromMessages")
        }

        @Test func mergeInstructionsBothNil() {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            #expect(client.mergeInstructions(nil) == nil)
        }

        @Test func generateRejectsMalformedHistory() async {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            let malformedHistory: [ChatMessage] = [
                .user("Hi"),
                .assistant(AssistantMessage(
                    content: "",
                    toolCalls: [ToolCall(id: "call_1", name: "lookup", arguments: "{}")]
                )),
            ]

            await #expect(throws: AgentError.malformedHistory(.unfinishedToolCallBatch(ids: ["call_1"]))) {
                _ = try await client.generate(
                    messages: malformedHistory,
                    tools: [],
                    responseFormat: nil,
                    requestContext: nil
                )
            }
        }

        @Test func streamRejectsMalformedHistory() async {
            guard #available(macOS 26, iOS 26, *) else { return }
            let client = FoundationModelsClient<EmptyContext>(context: EmptyContext())
            let malformedHistory: [ChatMessage] = [
                .user("Hi"),
                .assistant(AssistantMessage(
                    content: "",
                    toolCalls: [ToolCall(id: "call_1", name: "lookup", arguments: "{}")]
                )),
            ]

            await #expect(throws: AgentError.malformedHistory(.unfinishedToolCallBatch(ids: ["call_1"]))) {
                for try await _ in client.stream(messages: malformedHistory, tools: [], requestContext: nil) {}
            }
        }
    }

    private struct DummySchema: SchemaProviding, Codable {
        static let jsonSchema = JSONSchema.object(properties: [:], required: [])
    }

#endif
