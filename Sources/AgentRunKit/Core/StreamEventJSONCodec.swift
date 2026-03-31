import Foundation

/// Encodes and decodes canonical transcript JSON for streamed events.
public enum StreamEventJSONCodec {
    /// Returns the canonical encoder for transcript-grade event JSON.
    public static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.keyEncodingStrategy = .useDefaultKeys
        encoder.dataEncodingStrategy = .base64
        return encoder
    }

    /// Returns the canonical decoder for transcript-grade event JSON.
    public static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        decoder.dataDecodingStrategy = .base64
        return decoder
    }

    /// Encodes an event using the canonical transcript JSON configuration.
    public static func encode(_ event: StreamEvent) throws -> Data {
        try makeEncoder().encode(event)
    }

    /// Decodes an event using the canonical transcript JSON configuration.
    public static func decode(_ data: Data) throws -> StreamEvent {
        try makeDecoder().decode(StreamEvent.self, from: data)
    }
}
