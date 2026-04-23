@testable import AgentRunKit
import Foundation
import Testing

enum StreamEventInvariantAssertions {
    static func assertStage1RuntimeInvariants(
        _ events: [StreamEvent],
        startedAt: Date,
        endedAt: Date
    ) {
        var ids = Set<EventID>()
        for event in events {
            assertStage1RuntimeInvariants(
                event,
                startedAt: startedAt,
                endedAt: endedAt,
                ids: &ids
            )
        }
    }

    private static func assertStage1RuntimeInvariants(
        _ event: StreamEvent,
        startedAt: Date,
        endedAt: Date,
        ids: inout Set<EventID>
    ) {
        #expect(ids.insert(event.id).inserted)
        #expect(event.timestamp >= startedAt)
        #expect(event.timestamp <= endedAt)
        #expect(event.sessionID == nil)
        #expect(event.runID == nil)
        #expect(event.parentEventID == nil)
        #expect(event.origin == .live)
        if case let .subAgentEvent(_, _, nestedEvent) = event.kind {
            assertStage1RuntimeInvariants(
                nestedEvent,
                startedAt: startedAt,
                endedAt: endedAt,
                ids: &ids
            )
        }
    }
}
