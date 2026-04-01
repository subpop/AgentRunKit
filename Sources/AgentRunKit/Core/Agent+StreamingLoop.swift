import Foundation

struct StreamingLoopState {
    var messages: [ChatMessage]
    var budgetPhase: ContextBudgetPhase?
    var sessionAllowlist: Set<String> = []
}

extension Agent {
    func compactStreamingMessagesIfNeeded(
        _ messages: inout [ChatMessage],
        totalUsage: inout TokenUsage,
        lastTotalTokens: Int?,
        compactor: inout ContextCompactor,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) async {
        let compactionOutcome = await compactor.compactOrTruncateIfNeeded(
            &messages,
            lastTotalTokens: lastTotalTokens,
            totalUsage: &totalUsage
        )
        emitCompactionEventIfNeeded(
            compactionOutcome.emitsCompactionEvent,
            lastTotalTokens: lastTotalTokens,
            continuation: continuation
        )
    }

    func finalizeStreamingIteration(
        toolCalls: [ToolCall],
        context: C,
        budgetUsage: TokenUsage?,
        options: InvocationOptions,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation,
        state: inout StreamingLoopState
    ) async throws {
        let indexedCalls = indexedExecutableToolCalls(from: toolCalls)
        let pruneCalls = indexedCalls.filter { $0.call.name == "prune_context" }
        let regularCalls = indexedCalls.filter { $0.call.name != "prune_context" }

        let pruneOutcome = executePruneCalls(
            pruneCalls,
            messages: &state.messages,
            continuation: continuation
        )
        let regularResults = try await executeStreamingResults(
            regularCalls,
            context: context,
            messages: state.messages,
            continuation: continuation,
            approvalHandler: options.approvalHandler,
            allowlist: &state.sessionAllowlist
        )
        appendToolResults(
            (pruneOutcome.results + regularResults).sorted { $0.index < $1.index },
            messages: &state.messages
        )

        if let budgetUsage {
            applyBudgetPhase(
                &state.budgetPhase,
                usage: budgetUsage,
                messages: &state.messages,
                continuation: continuation
            )
        }
        try state.messages.validateForAgentHistory()
    }
}
