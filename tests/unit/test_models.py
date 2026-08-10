from toolwarden.models import ToolCallRequest, ToolCallResult


def test_tool_call_request_to_dict_has_expected_shape():
    request = ToolCallRequest(
        tool_name="get_time",
        arguments={"timezone": "Europe/Dublin"},
        source="openai",
        session_id="sess-1",
    )
    record = request.to_dict()

    assert record["event"] == "tool_call_request"
    assert record["tool_name"] == "get_time"
    assert record["arguments"] == {"timezone": "Europe/Dublin"}
    assert record["session_id"] == "sess-1"
    assert record["call_id"] == request.call_id


def test_tool_call_request_generates_unique_call_ids():
    a = ToolCallRequest(tool_name="t", arguments={}, source="openai", session_id="s")
    b = ToolCallRequest(tool_name="t", arguments={}, source="openai", session_id="s")

    assert a.call_id != b.call_id


def test_tool_call_result_to_dict_has_expected_shape():
    result = ToolCallResult(
        call_id="call-1",
        tool_name="get_time",
        content="12:00",
        is_error=False,
        latency_ms=5,
    )
    record = result.to_dict()

    assert record["event"] == "tool_call_result"
    assert record["call_id"] == "call-1"
    assert record["content"] == "12:00"
    assert record["is_error"] is False
    assert record["latency_ms"] == 5
