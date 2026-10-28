"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.config import config


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg() -> None:
    """The tool base64-encodes the JPEG bytes returned by the SDK."""
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert result["b64_im"] == base64.b64encode(jpeg_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame() -> None:
    """With no frame available the tool returns an error."""
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = None

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error and never reads a frame."""
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=False,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_camera_tool_xai_payload_and_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    """In xAI mode the tool should send the correct chat-completions image_url+text parts and return the caption."""
    # Configure xAI mode and key
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "xai")
    monkeypatch.setattr(config, "XAI_API_KEY", "sk-xai")
    monkeypatch.setattr(config, "XAI_BASE_URL", "https://api.x.ai/v1")
    monkeypatch.setattr(config, "XAI_VISION_MODEL", "grok-4.7")

    sent: dict = {}

    class FakeResponse:
        status_code = 200

        def __init__(self, json_payload: dict) -> None:
            self._json = json_payload
            self.text = "{}"

        @property
        def is_error(self) -> bool:
            return False

        def json(self) -> dict:
            return self._json

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args) -> bool:
            return False

        async def post(self, path: str, headers: dict, json: dict) -> FakeResponse:
            sent["path"] = path
            sent["headers"] = headers
            sent["json"] = json
            return FakeResponse(
                {
                    "choices": [
                        {"message": {"content": [{"type": "text", "text": "a blue mug on the table"}]}}
                    ]
                }
            )

    monkeypatch.setattr("reachy_mini_conversation_app.tools.camera.httpx.AsyncClient", FakeClient)

    # Simulate a JPEG frame
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes
    deps = ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock(), camera_enabled=True)

    result = await Camera()(deps, question="What do you see?")

    assert result == {"caption": "a blue mug on the table"}
    assert sent["path"] == "/chat/completions"
    assert sent["json"]["model"] == "grok-4.7"
    content = sent["json"]["messages"][0]["content"]
    # Expect first part image_url with data URL and detail=high, then text
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[0]["image_url"]["detail"] == "high"
    assert content[1] == {"type": "text", "text": "What do you see?"}
