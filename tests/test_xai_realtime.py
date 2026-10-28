"""xAI (Grok Voice) mode tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock

import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.streaming import AdditionalOutputs


def _plain_handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


class _FakeEvent:
    def __init__(self, event_type: str, **fields):
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(*, events: tuple[_FakeEvent, ...] = ()) -> any:
    class FakeSession:
        async def update(self, **_kwargs):
            pass

    class FakeNoop:
        async def append(self, **_kw): ...
        async def create(self, **_kw): ...
        async def cancel(self, **_kw): ...

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self): self._events = iter(events)
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return False
        def __aiter__(self): return self
        async def __anext__(self):
            try: return next(self._events)
            except StopIteration: raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **_kwargs): return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def test_xai_connection_selection_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI mode reports a usable target only when the API key is set."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "xai")
    monkeypatch.setattr(config, "XAI_API_KEY", "")
    sel = hf_mod.get_hf_connection_selection()
    assert sel.mode == "xai" and sel.has_target is False
    monkeypatch.setattr(config, "XAI_API_KEY", "sk-xai")
    sel = hf_mod.get_hf_connection_selection()
    assert sel.mode == "xai" and sel.has_target is True


def test_xai_session_payload_uses_16khz_and_no_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI sessions set an explicit 16 kHz PCM rate and do not send a transcription model."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "xai")
    handler = _plain_handler()
    session = handler._get_session_config([])
    assert session["audio"]["input"]["format"]["rate"] == 16000
    assert session["audio"]["output"]["format"]["rate"] == 16000
    assert session["audio"]["input"].get("transcription") in (None, {})


@pytest.mark.asyncio
async def test_xai_transcription_updated_emits_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    """xAI updated snapshots produce a debounced user partial transcript."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "xai")
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default="eve": "eve")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    handler = _plain_handler()
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.updated", item_id="i1", transcript="Hello"),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    handler.partial_debounce_delay = 0
    transcripts: list[tuple[str, str, bool]] = []
    handler.set_transcript_observer(lambda role, text, final: transcripts.append((role, text, final)))
    await handler._run_realtime_session()
    if handler.partial_transcript_task is not None:
        await handler.partial_transcript_task
    # The debounced partial is pushed to both the queue and the transcript observer.
    assert transcripts == [("user", "Hello", False)]
    messages = [item.args[0] for item in _drain(handler) if isinstance(item, AdditionalOutputs)]
    assert {"role": "user_partial", "content": "Hello"} in messages


def _drain(handler: HuggingFaceRealtimeHandler):
    items = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items

