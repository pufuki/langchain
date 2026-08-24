"""Structured exceptions for LangChain agents."""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import ErrorCode, LangChainException, create_message


class AgentError(LangChainException):
    """Base class for all agent-related errors."""

    error_code: ErrorCode = ErrorCode.AGENT_ERROR

    def __init__(self, message: str, *, error_code: ErrorCode | None = None) -> None:
        """Initialize an AgentError.

        Args:
            message: The error message.
            error_code: Optional error code. Defaults to class-level error_code.
        """
        code = error_code or self.error_code
        super().__init__(create_message(message=message, error_code=code))
        self.error_code = code


class MiddlewareError(AgentError):
    """Error originating from middleware execution.

    This exception wraps errors that occur during middleware hook execution,
    providing context about which middleware and hook failed.
    """

    error_code: ErrorCode = ErrorCode.AGENT_MIDDLEWARE_ERROR

    def __init__(
        self,
        middleware_name: str,
        hook: str,
        original_error: Exception,
        request_context: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize a MiddlewareError.

        Args:
            middleware_name: Name of the middleware that raised the error.
            hook: The middleware hook that failed (e.g., "wrap_model_call",
                "wrap_tool_call", "before_agent", "after_agent").
            original_error: The original exception that was raised.
            request_context: Sanitized request context for debugging
                (tool name, arg keys, call ID, etc.).
            suggestion: Optional actionable suggestion for fixing the error.
        """
        self.middleware_name = middleware_name
        self.hook = hook
        self.original_error = original_error
        self.request_context = request_context or {}
        self.suggestion = suggestion

        parts = [
            f"Middleware '{middleware_name}' failed in hook '{hook}': {original_error}",
        ]
        if request_context:
            context_str = ", ".join(f"{k}={v}" for k, v in request_context.items())
            parts.append(f"Context: {context_str}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        message = " ".join(parts)
        super().__init__(message, error_code=self.error_code)


class ToolExecutionError(AgentError):
    """Tool execution failed with actionable guidance.

    This exception provides detailed context about tool failures including
    the tool name, arguments, schema, and a suggestion for resolution.
    """

    error_code: ErrorCode = ErrorCode.AGENT_TOOL_EXECUTION_ERROR

    def __init__(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_schema: dict[str, Any] | None,
        original_error: Exception,
        suggestion: str,
        *,
        is_retryable: bool = False,
    ) -> None:
        """Initialize a ToolExecutionError.

        Args:
            tool_name: Name of the tool that failed.
            tool_args: Arguments passed to the tool.
            tool_schema: The tool's argument schema (if available).
            original_error: The original exception raised by the tool.
            suggestion: Actionable suggestion for fixing the error.
            is_retryable: Whether the error is retryable (e.g., transient failure).
        """
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_schema = tool_schema
        self.original_error = original_error
        self.suggestion = suggestion
        self.is_retryable = is_retryable

        parts = [
            f"Tool '{tool_name}' execution failed: {original_error}",
            f"Suggestion: {suggestion}",
        ]
        if tool_schema:
            parts.append(f"Expected schema: {tool_schema.get('properties', tool_schema)}")

        message = " ".join(parts)
        super().__init__(message, error_code=self.error_code)


class ModelCallError(AgentError):
    """Model invocation failed with retry context.

    This exception wraps model invocation failures and provides context
    about retry attempts and suggestions for resolution.
    """

    error_code: ErrorCode = ErrorCode.AGENT_MODEL_CALL_ERROR

    def __init__(
        self,
        model_name: str,
        attempt: int,
        max_retries: int,
        original_error: Exception,
        suggestion: str,
    ) -> None:
        """Initialize a ModelCallError.

        Args:
            model_name: Name of the model that failed.
            attempt: The attempt number that failed (1-indexed).
            max_retries: Maximum number of retries configured.
            original_error: The original exception from the model call.
            suggestion: Actionable suggestion for fixing the error.
        """
        self.model_name = model_name
        self.attempt = attempt
        self.max_retries = max_retries
        self.original_error = original_error
        self.suggestion = suggestion

        parts = [
            (
                f"Model '{model_name}' call failed on attempt "
                f"{attempt}/{max_retries + 1}: {original_error}"
            ),
            f"Suggestion: {suggestion}",
        ]

        message = " ".join(parts)
        super().__init__(message, error_code=self.error_code)


class AgentConfigurationError(AgentError):
    """Invalid agent configuration.

    Raised when agent creation fails due to invalid configuration such as
    duplicate middleware, missing required tools, or incompatible settings.
    """

    error_code: ErrorCode = ErrorCode.AGENT_CONFIGURATION_ERROR

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        """Initialize an AgentConfigurationError.

        Args:
            message: Description of the configuration error.
            suggestion: Optional actionable suggestion for fixing the configuration.
        """
        self.suggestion = suggestion
        parts = [message]
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")
        super().__init__(" ".join(parts), error_code=self.error_code)


class GraphExecutionError(AgentError):
    """Wrapper for LangGraph GraphBubbleUp with agent context.

    This exception wraps LangGraph's GraphBubbleUp control-flow signals
    to provide agent-specific context about which node and state caused
    the bubble-up.
    """

    error_code: ErrorCode = ErrorCode.AGENT_GRAPH_EXECUTION_ERROR

    def __init__(
        self,
        node_name: str,
        state_snapshot: dict[str, Any],
        original_error: Exception,
    ) -> None:
        """Initialize a GraphExecutionError.

        Args:
            node_name: Name of the graph node where the error originated.
            state_snapshot: Sanitized snapshot of the agent state at the time of error.
            original_error: The original GraphBubbleUp or other exception.
        """
        self.node_name = node_name
        self.state_snapshot = state_snapshot
        self.original_error = original_error

        # Sanitize state snapshot for display
        safe_state = {k: v for k, v in state_snapshot.items() if k not in ("runtime", "handler")}
        message = (
            f"Graph execution error at node '{node_name}': {original_error}. "
            f"State keys: {list(safe_state.keys())}"
        )
        super().__init__(message, error_code=self.error_code)


def sanitize_tool_request(request: Any) -> dict[str, Any]:
    """Extract safe debugging info from a tool call request.

    Args:
        request: The tool call request object (ToolCallRequest or similar).

    Returns:
        Dictionary with safe debugging information (no sensitive data).
    """
    if not hasattr(request, "tool_call"):
        return {}

    tool_call = request.tool_call
    return {
        "tool_name": tool_call.get("name"),
        "tool_args_keys": list(tool_call.get("args", {}).keys()),
        "call_id": tool_call.get("id"),
    }


def sanitize_model_request(request: Any) -> dict[str, Any]:
    """Extract safe debugging info from a model request.

    Args:
        request: The model request object (ModelRequest or similar).

    Returns:
        Dictionary with safe debugging information.
    """
    result: dict[str, Any] = {}
    if hasattr(request, "messages"):
        result["message_count"] = len(request.messages)
        result["last_message_type"] = (
            type(request.messages[-1]).__name__ if request.messages else None
        )
    if hasattr(request, "tools"):
        result["tool_count"] = len(request.tools)
        result["tool_names"] = [getattr(t, "name", str(t)) for t in request.tools]
    if hasattr(request, "system_message") and request.system_message:
        result["has_system_message"] = True
    return result
