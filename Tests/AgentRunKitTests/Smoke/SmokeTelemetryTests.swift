@testable import AgentRunKit
import Foundation
import Testing

struct SmokeTelemetryTests {
    @Test func classifierRecognizesAssertionFailure() {
        let classification = classifySmokeFailure(
            SmokeAssertionFailure(fileID: "Tests/Foo.swift", line: 12, message: "Expected success")
        )

        #expect(classification.kind == .assertionFailure)
        #expect(classification.bodyExcerpt?.contains("Expected success") == true)
    }

    @Test func classifierRecognizesHTTPError() {
        let classification = classifySmokeFailure(
            AgentError.llmError(.httpError(statusCode: 400, body: "invalid_request_error"))
        )

        #expect(classification.kind == .httpError)
        #expect(classification.httpStatus == 400)
        #expect(classification.bodyExcerpt?.contains("invalid_request_error") == true)
    }

    @Test func classifierCapturesStructuredOutputRawText() {
        let classification = classifySmokeFailure(
            SmokeStructuredOutputFailure(
                rawContent: "partial { exercises: [",
                underlyingDescription: "malformed JSON"
            )
        )

        #expect(classification.kind == .structuredOutputDecodingFailed)
        #expect(classification.assistantTextExcerpt?.contains("partial") == true)
    }

    @Test func jsonlWritesParseableLine() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("smoke-\(UUID().uuidString).jsonl").path
        defer {
            try? FileManager.default.removeItem(atPath: path)
        }

        try appendSmokeJSONL(
            SmokeTelemetryRecord(
                suite: "suite",
                test: "test",
                provider: "provider",
                model: "model",
                durationMillis: 12,
                kind: .httpError,
                httpStatus: 500,
                bodyExcerpt: "body",
                assistantTextExcerpt: nil
            ),
            to: path
        )

        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let contents = try #require(String(data: data, encoding: .utf8))
        let lines = contents.split(separator: "\n")

        #expect(lines.count == 1)

        let decoded = try JSONDecoder().decode(SmokeTelemetryRecord.self, from: Data(lines[0].utf8))
        #expect(decoded.suite == "suite")
        #expect(decoded.kind == .httpError)
        #expect(decoded.httpStatus == 500)
    }
}
