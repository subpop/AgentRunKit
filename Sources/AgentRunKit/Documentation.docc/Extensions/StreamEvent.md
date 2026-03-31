# ``AgentRunKit/StreamEvent``

Stable envelope for a streamed event.

Each event carries identity and timing metadata around a semantic ``StreamEvent/Kind`` payload. Transcript order is the order of emission, not the timestamp sort order. Direct `Agent` and `Chat` streams leave `sessionID`, `runID`, and `parentEventID` unset until the session layer owns event construction.

## Topics

### Envelope

- ``init(id:timestamp:sessionID:runID:parentEventID:kind:)``
- ``id``
- ``timestamp``
- ``sessionID``
- ``runID``
- ``parentEventID``
- ``kind``

### Semantic Payload

- ``StreamEvent/Kind``
