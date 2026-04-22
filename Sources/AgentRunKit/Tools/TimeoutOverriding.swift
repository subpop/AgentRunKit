import Foundation

protocol TimeoutOverriding {
    /// nil means "inherit the agent or chat's configured timeout", not "no timeout".
    var toolTimeout: Duration? { get }
}

func resolvedToolTimeout<C: ToolContext>(
    for tool: any AnyTool<C>,
    default fallback: Duration
) -> Duration {
    (tool as? any TimeoutOverriding)?.toolTimeout ?? fallback
}
