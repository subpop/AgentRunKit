import Foundation

struct ToolCallAccumulator {
    let id: String
    let name: String
    var arguments: String = ""

    func toToolCall() -> ToolCall {
        ToolCall(id: id, name: name, arguments: arguments.isEmpty ? "{}" : arguments)
    }
}
