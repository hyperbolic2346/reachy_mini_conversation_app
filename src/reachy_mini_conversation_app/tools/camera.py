import base64
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.config import HF_XAI_CONNECTION_MODE, config
import httpx


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live, each call captures the current moment. "
        "If the user asks you to look without saying at what, do not ask for clarification, call this tool and describe what you see. "
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to observe or ask about in the picture. "
                    "Examples: what is the user holding, describe the user's outfit, "
                    "what do you see around you, how does the user look today."
                ),
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and return the base64-encoded JPEG."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        # In xAI mode, caption the image via chat completions instead of attaching it to the realtime session.
        if getattr(config, "HF_REALTIME_CONNECTION_MODE", None) == HF_XAI_CONNECTION_MODE:
            api_key = (getattr(config, "XAI_API_KEY", None) or "").strip()
            if not api_key:
                logger.error("XAI_API_KEY not configured; cannot caption image")
                return {"error": "xAI API key not configured"}
            base_url = (getattr(config, "XAI_BASE_URL", None) or "https://api.x.ai/v1").strip().rstrip("/")
            vision_model = (getattr(config, "XAI_VISION_MODEL", None) or "grok-4.7").strip()
            b64_im = base64.b64encode(jpeg_bytes).decode("utf-8")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload: Dict[str, Any] = {
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_im}", "detail": "high"},
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            }
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
                    resp = await client.post("/chat/completions", headers=headers, json=payload)
                    if resp.is_error:
                        body = resp.text
                        snippet = (body[:500] + "…") if len(body) > 500 else body
                        logger.warning(
                            "xAI captioning HTTP %s: %s",
                            getattr(resp, "status_code", "unknown"),
                            snippet,
                        )
                        return {"error": f"xAI captioning failed: HTTP {resp.status_code}"}
                    data = resp.json()
            except httpx.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                body = getattr(getattr(exc, "response", None), "text", "")
                snippet = (body[:500] + "…") if isinstance(body, str) and len(body) > 500 else body
                logger.warning("xAI captioning HTTP error%s: %s", f" {status}" if status else "", snippet or exc)
                return {"error": f"xAI captioning failed{f' ({status})' if status else ''}"}
            # Extract the best-effort text from OpenAI-compatible response
            try:
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    caption = content.strip()
                elif isinstance(content, list):
                    # Search for a text segment
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                            text = (part.get("text") or "").strip()
                            if text:
                                text_parts.append(text)
                    caption = " ".join(text_parts).strip()
                else:
                    caption = ""
                if not caption:
                    logger.warning("xAI captioning returned no content")
                    return {"error": "xAI captioning returned no content"}
                return {"caption": caption}
            except Exception as exc:
                logger.warning("xAI caption parsing failed: %s", exc)
                return {"error": "Failed to parse xAI caption response"}

        # Default HF mode: return the image so the realtime session can attach it.
        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
