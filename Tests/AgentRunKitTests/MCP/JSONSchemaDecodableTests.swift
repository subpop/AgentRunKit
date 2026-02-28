import Foundation
import Testing

@testable import AgentRunKit

@Suite
struct JSONSchemaDecodableTests {
    private func roundTrip(_ schema: JSONSchema) throws -> JSONSchema {
        let data = try JSONEncoder().encode(schema)
        return try JSONDecoder().decode(JSONSchema.self, from: data)
    }

    @Test
    func stringSchemaRoundTrip() throws {
        let original = JSONSchema.string(description: "A name", enumValues: ["a", "b"])
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func integerSchemaRoundTrip() throws {
        let original = JSONSchema.integer(description: "Count")
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func numberSchemaRoundTrip() throws {
        let original = JSONSchema.number(description: "Price")
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func booleanSchemaRoundTrip() throws {
        let original = JSONSchema.boolean(description: "Flag")
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func nullSchemaRoundTrip() throws {
        let original = JSONSchema.null
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func arraySchemaRoundTrip() throws {
        let original = JSONSchema.array(items: .integer(), description: "Numbers")
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func objectSchemaRoundTrip() throws {
        let original = JSONSchema.object(
            properties: [
                "name": .string(description: "User name"),
                "age": .integer(),
            ],
            required: ["name"],
            description: "A user"
        )
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func anyOfSchemaRoundTrip() throws {
        let original = JSONSchema.anyOf([.string(), .null])
        let decoded = try roundTrip(original)
        #expect(decoded == original)
    }

    @Test
    func unknownTypeThrows() {
        let json = #"{"type":"unknown"}"#
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(JSONSchema.self, from: Data(json.utf8))
        }
    }

    @Test
    func missingTypeDecodesAsEmptyObject() throws {
        let json = #"{}"#
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: Data(json.utf8))
        #expect(decoded == .object(properties: [:], required: []))
    }
}
