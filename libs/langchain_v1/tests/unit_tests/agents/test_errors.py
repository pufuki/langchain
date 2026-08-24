"""Tests for agent exceptions."""

from __future__ import annotations

from typing import Any

from langchain_core.exceptions import ErrorCode

from langchain.agents.errors import (
    AgentConfigurationError,
    AgentError,
    GraphExecutionError,
    MiddlewareError,
    ModelCallError,
    ToolExecutionError,
    sanitize_model_request,
    sanitize_tool_request,
)


class TestAgentError:
    """Tests for AgentError base class."""

    def test_agent_error_default_code(self) -> None:
        """Test AgentError uses default error code."""
        err = AgentError("Something went wrong")
        assert err.error_code == ErrorCode.AGENT_ERROR
        assert "Something went wrong" in str(err)
        assert "troubleshooting" in str(err).lower()

    def test_agent_error_custom_code(self) -> None:
        """Test AgentError accepts custom error code."""
        err = AgentError("Custom error", error_code=ErrorCode.AGENT_CONFIGURATION_ERROR)
        assert err.error_code == ErrorCode.AGENT_CONFIGURATION_ERROR


class TestMiddlewareError:
    """Tests for MiddlewareError."""

    def test_middleware_error_basic(self) -> None:
        """Test MiddlewareError with required fields."""
        original = ValueError("Invalid input")
        err = MiddlewareError(
            middleware_name="ToolRetryMiddleware",
            hook="wrap_tool_call",
            original_error=original,
        )

        assert err.middleware_name == "ToolRetryMiddleware"
        assert err.hook == "wrap_tool_call"
        assert err.original_error is original
        assert err.request_context == {}
        assert err.suggestion is None
        assert err.error_code == ErrorCode.AGENT_MIDDLEWARE_ERROR
        assert "ToolRetryMiddleware" in str(err)
        assert "wrap_tool_call" in str(err)
        assert "Invalid input" in str(err)

    def test_middleware_error_with_context_and_suggestion(self) -> None:
        """Test MiddlewareError with context and suggestion."""
        original = ConnectionError("Connection refused")
        err = MiddlewareError(
            middleware_name="ModelFallbackMiddleware",
            hook="wrap_model_call",
            original_error=original,
            request_context={"model": "gpt-4", "attempt": "2"},
            suggestion="Check API key and network connectivity",
        )

        assert err.request_context == {"model": "gpt-4", "attempt": "2"}
        assert err.suggestion == "Check API key and network connectivity"
        assert "gpt-4" in str(err)
        assert "API key" in str(err)

    def test_middleware_error_inheritance(self) -> None:
        """Test MiddlewareError is an AgentError."""
        err = MiddlewareError("test", "hook", ValueError("x"))
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_tool_execution_error_basic(self) -> None:
        """Test ToolExecutionError with required fields."""
        original = RuntimeError("Tool crashed")
        err = ToolExecutionError(
            tool_name="search_web",
            tool_args={"query": "test"},
            tool_schema={"properties": {"query": {"type": "string"}}},
            original_error=original,
            suggestion="Check tool implementation",
            is_retryable=True,
        )

        assert err.tool_name == "search_web"
        assert err.tool_args == {"query": "test"}
        assert err.tool_schema == {"properties": {"query": {"type": "string"}}}
        assert err.original_error is original
        assert err.suggestion == "Check tool implementation"
        assert err.is_retryable is True
        assert err.error_code == ErrorCode.AGENT_TOOL_EXECUTION_ERROR
        assert "search_web" in str(err)
        assert "Tool crashed" in str(err)
        assert "Check tool implementation" in str(err)
        assert "Expected schema" in str(err)

    def test_tool_execution_error_without_schema(self) -> None:
        """Test ToolExecutionError without schema."""
        err = ToolExecutionError(
            tool_name="simple_tool",
            tool_args={},
            tool_schema=None,
            original_error=ValueError("x"),
            suggestion="Try again",
        )
        assert err.tool_schema is None
        assert "Expected schema" not in str(err)

    def test_tool_execution_error_inheritance(self) -> None:
        """Test ToolExecutionError is an AgentError."""
        err = ToolExecutionError("t", {}, None, ValueError("x"), "s")
        assert isinstance(err, AgentError)


class TestModelCallError:
    """Tests for ModelCallError."""

    def test_model_call_error_basic(self) -> None:
        """Test ModelCallError with required fields."""
        original = TimeoutError("Request timed out")
        err = ModelCallError(
            model_name="gpt-4",
            attempt=2,
            max_retries=3,
            original_error=original,
            suggestion="Increase timeout or check network",
        )

        assert err.model_name == "gpt-4"
        assert err.attempt == 2
        assert err.max_retries == 3
        assert err.original_error is original
        assert err.suggestion == "Increase timeout or check network"
        assert err.error_code == ErrorCode.AGENT_MODEL_CALL_ERROR
        assert "gpt-4" in str(err)
        assert "attempt 2/4" in str(err)
        assert "Increase timeout" in str(err)

    def test_model_call_error_inheritance(self) -> None:
        """Test ModelCallError is an AgentError."""
        err = ModelCallError("m", 1, 1, ValueError("x"), "s")
        assert isinstance(err, AgentError)


class TestAgentConfigurationError:
    """Tests for AgentConfigurationError."""

    def test_configuration_error_basic(self) -> None:
        """Test AgentConfigurationError with message only."""
        err = AgentConfigurationError("Duplicate middleware")
        assert err.suggestion is None
        assert "Duplicate middleware" in str(err)
        assert "troubleshooting" in str(err).lower()
        assert err.error_code == ErrorCode.AGENT_CONFIGURATION_ERROR

    def test_configuration_error_with_suggestion(self) -> None:
        """Test AgentConfigurationError with suggestion."""
        err = AgentConfigurationError("No tools provided", suggestion="Add at least one tool")
        assert err.suggestion == "Add at least one tool"
        assert "No tools provided" in str(err)
        assert "Add at least one tool" in str(err)

    def test_configuration_error_inheritance(self) -> None:
        """Test AgentConfigurationError is an AgentError."""
        err = AgentConfigurationError("msg")
        assert isinstance(err, AgentError)


class TestGraphExecutionError:
    """Tests for GraphExecutionError."""

    def test_graph_execution_error_basic(self) -> None:
        """Test GraphExecutionError with required fields."""
        original = Exception("Graph bubble up")
        state = {"messages": [], "runtime": "should_be_filtered", "custom_key": "value"}
        err = GraphExecutionError(
            node_name="model_node",
            state_snapshot=state,
            original_error=original,
        )

        assert err.node_name == "model_node"
        assert err.state_snapshot == state
        assert err.original_error is original
        assert err.error_code == ErrorCode.AGENT_GRAPH_EXECUTION_ERROR
        assert "model_node" in str(err)
        assert "Graph bubble up" in str(err)
        assert "State keys:" in str(err)
        # runtime should be filtered from display
        assert "runtime" not in str(err)
        assert "custom_key" in str(err)

    def test_graph_execution_error_inheritance(self) -> None:
        """Test GraphExecutionError is an AgentError."""
        err = GraphExecutionError("node", {}, ValueError("x"))
        assert isinstance(err, AgentError)


class TestSanitizeFunctions:
    """Tests for sanitization helpers."""

    def test_sanitize_tool_request(self) -> None:
        """Test sanitize_tool_request extracts safe info."""

        class MockRequest:
            def __init__(self) -> None:
                self.tool_call = {
                    "name": "search",
                    "args": {"query": "secret", "api_key": "hidden"},
                    "id": "call_123",
                }

        result = sanitize_tool_request(MockRequest())
        assert result == {
            "tool_name": "search",
            "tool_args_keys": ["query", "api_key"],
            "call_id": "call_123",
        }

    def test_sanitize_tool_request_missing_tool_call(self) -> None:
        """Test sanitize_tool_request handles missing tool_call."""

        class MockRequest:
            pass

        result = sanitize_tool_request(MockRequest())
        assert result == {}

    def test_sanitize_model_request(self) -> None:
        """Test sanitize_model_request extracts safe info."""

        class Tool:
            name = "tool1"

        class Tool2:
            name = "tool2"

        class MockRequest:
            def __init__(self) -> None:
                self.messages: list[str] = ["msg1", "msg2"]
                self.tools: list[Any] = [Tool(), Tool2()]
                self.system_message = "System prompt"

        result = sanitize_model_request(MockRequest())
        assert result == {
            "message_count": 2,
            "last_message_type": "str",
            "tool_count": 2,
            "tool_names": ["tool1", "tool2"],
            "has_system_message": True,
        }

    def test_sanitize_model_request_minimal(self) -> None:
        """Test sanitize_model_request with minimal request."""

        class MockRequest:
            def __init__(self) -> None:
                self.messages: list[str] = []
                self.tools: list[Any] = []

        result = sanitize_model_request(MockRequest())
        assert result == {
            "message_count": 0,
            "last_message_type": None,
            "tool_count": 0,
            "tool_names": [],
        }
