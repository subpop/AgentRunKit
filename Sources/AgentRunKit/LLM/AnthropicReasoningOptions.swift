import Foundation

/// Provider-specific request options for Anthropic reasoning behavior.
public struct AnthropicReasoningOptions: Sendable, Equatable {
    /// The Anthropic wire-level strategy used to lower shared reasoning intent.
    public enum Mode: Sendable, Equatable {
        case manual
        case adaptive
    }

    public let mode: Mode

    public init(mode: Mode = .manual) {
        self.mode = mode
    }

    public static let manual = AnthropicReasoningOptions()
    public static let adaptive = AnthropicReasoningOptions(mode: .adaptive)
}
