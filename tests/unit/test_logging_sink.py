from toolwarden.logging_sink import InMemorySink, JsonlFileSink
import json


def test_in_memory_sink_collects_records_in_order():
    sink = InMemorySink()
    sink.write({"event": "a"})
    sink.write({"event": "b"})

    assert [r["event"] for r in sink.records] == ["a", "b"]


def test_jsonl_file_sink_appends_one_json_object_per_line(tmp_path):
    log_path = tmp_path / "nested" / "traffic.jsonl"
    sink = JsonlFileSink(log_path)

    sink.write({"event": "tool_call_request", "tool_name": "get_time"})
    sink.write({"event": "tool_call_result", "tool_name": "get_time"})

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "tool_call_request"
    assert json.loads(lines[1])["event"] == "tool_call_result"
