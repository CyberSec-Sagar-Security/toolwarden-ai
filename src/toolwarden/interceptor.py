"""Phase 2: passthrough only.

Records every outbound tool-call request and inbound tool-call result,
blocks nothing. This proves the capture point actually sees full traffic
before any classification/enforcement logic is added (Phase 4, Phase 6).
"""

from __future__ import annotations

from toolwarden.logging_sink import LogSink
from toolwarden.models import ToolCallRequest, ToolCallResult


class Interceptor:
    def __init__(self, sink: LogSink) -> None:
        self.sink = sink

    def intercept_request(self, request: ToolCallRequest) -> ToolCallRequest:
        self.sink.write(request.to_dict())
        return request

    def intercept_result(self, result: ToolCallResult) -> ToolCallResult:
        self.sink.write(result.to_dict())
        return result
