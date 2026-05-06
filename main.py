"""
OpenAI 兼容节点（与仓库 test.py 用法对齐）：
- `images`：`client.images.generate` / `images.edit`（同 test.py：先带 response_format=b64_json，失败再重试）
  默认网关示例：https://uclaude.cc/v1 + gpt-image-2
- `chat_completions`：`/v1/chat/completions` 多模态（识图请换支持 vision 的模型，勿用 gpt-image-2）

输出：image、text、request、response（request 为等价 curl 预览，密钥占位；大图 base64 已截断）
可选参考图：image1～image10（最多 10 张）；chat 多图入消息；images 编辑模式按序上传多图（依网关与模型能力）。
"""

import base64
import io
import json
import os
import shlex
import time
import traceback
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from openai import APIStatusError, OpenAI


_RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_TRANSIENT_RETRIES = 4


def _call_with_transient_retry(fn):
    """网关/上游瞬时错误（502、503 等）时有限次退避重试。"""
    for attempt in range(_MAX_TRANSIENT_RETRIES):
        try:
            return fn()
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code not in _RETRYABLE_HTTP_STATUS:
                raise
            if attempt >= _MAX_TRANSIENT_RETRIES - 1:
                raise
            time.sleep(min(2**attempt, 30.0))


def _normalize_base_url(url: str) -> str:
    """
    Cherry 风格若只填域名（无路径），补 /v1，与 test.py 里 BASE_URL=https://xxx/v1 一致。
    - api.openai.com、codeagent.cloud：裸域名 → .../v1
    其它带自定义路径的网关（Azure、/pricing 等）原样返回。
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    parsed = urllib.parse.urlparse(u)
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")
    scheme = parsed.scheme or "https"
    bare = path == "" or path == "/"
    if not bare:
        return u
    if netloc == "api.openai.com" or netloc.endswith(
        "codeagent.cloud"
    ):
        host = parsed.netloc or netloc
        return f"{scheme}://{host}/v1"
    return u


# 节点「尺寸」下拉文案 -> Images API 的 `size`（含 auto、具体 WxH）
_UI_SIZE_LABEL_TO_API: Dict[str, str] = {
    "自动（auto）": "auto",
    "1024x1024 - 1K（1:1方图）": "1024x1024",
    "1536x1024 - 1.5K（3:2横图）": "1536x1024",
    "1024x1536 - 1.5K（2:3竖图）": "1024x1536",
    "2048x2048 - 2K（1:1方图）": "2048x2048",
    "2880x2880 - 3K（1:1方图）": "2880x2880",
    "3840x2160 - 4K（16:9横图）": "3840x2160",
    "2160x3840 - 4K（9:16竖图）": "2160x3840",
}


def _resolve_images_api_size(ui_choice: str) -> str:
    """将界面上的尺寸选项解析为 API 使用的 size 字符串。"""
    k = (ui_choice or "").strip()
    if k in _UI_SIZE_LABEL_TO_API:
        return _UI_SIZE_LABEL_TO_API[k]
    return k


# 节点「质量」下拉文案 -> Images API 的 quality（None 表示不传）；须与 INPUT_TYPES 一致
_UI_QUALITY_LABEL_TO_API: Dict[str, Optional[str]] = {
    "自动（auto）": None,
    "低（low）": "low",
    "中（medium）": "medium",
    "高（high）": "high",
}


def _resolve_images_api_quality(ui_choice: str) -> Optional[str]:
    """将界面上的质量选项解析为 API 的 quality；仅认当前 INPUT_TYPES 中的文案。"""
    k = (ui_choice or "").strip()
    return _UI_QUALITY_LABEL_TO_API.get(k)


def _failed_run_outputs(exc: BaseException, req_preview: str) -> Tuple[torch.Tensor, str, str, str]:
    """出错时仍返回四路输出，便于从 request 查看等价请求。"""
    msg = f"{type(exc).__name__}: {exc}"
    req = (req_preview or "").strip() or "# （未能生成 request 预览）"
    detail = f"{msg}\n\n{traceback.format_exc()}"
    return (_blank_image(), msg, req, detail)


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    """Encode first batch image as PNG bytes."""
    tensor = image[0].clamp(0, 1)
    np_img = (tensor * 255).byte().cpu().numpy()
    pil_img = Image.fromarray(np_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def _tensor_to_image_data_uri(image: torch.Tensor) -> str:
    """OpenAI vision message: image_url url can be data:image/png;base64,..."""
    png = _tensor_to_png_bytes(image)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _blank_image() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _passthrough_or_blank(images: Any) -> torch.Tensor:
    if images is not None and isinstance(images, torch.Tensor) and images.numel() > 0:
        return images[0:1].clamp(0, 1).contiguous()
    return _blank_image()


def _collect_optional_images(
    *slots: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """按 image1、image2…顺序收集非空 IMAGE（每槽取 batch 首张）。"""
    out: List[torch.Tensor] = []
    for t in slots:
        if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
            out.append(t[0:1].clamp(0, 1).contiguous())
    return out


def _response_to_raw_string(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return json.dumps(resp, ensure_ascii=False, indent=2)
    model_dump_json = getattr(resp, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return model_dump_json()
        except TypeError:
            pass
    model_dump = getattr(resp, "model_dump", None)
    if callable(model_dump):
        try:
            return json.dumps(model_dump(), ensure_ascii=False, indent=2)
        except TypeError:
            pass
    return str(resp)


def _content_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    pieces.append(t)
            elif isinstance(part, str):
                pieces.append(part)
        return "\n".join(pieces)
    return ""


def _extract_text_from_choice(choice: Any) -> str:
    if choice is None:
        return ""
    if isinstance(choice, dict):
        msg = choice.get("message")
        if isinstance(msg, str):
            return msg
        if isinstance(msg, dict):
            return _content_from_message_content(msg.get("content"))
        return ""
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    content = getattr(msg, "content", "")
    return _content_from_message_content(content)


def _extract_text_from_completion(response: Any) -> str:
    """兼容网关返回 str / dict / SDK 对象（避免 .choices 报错）。"""
    if response is None:
        return ""
    if isinstance(response, str):
        s = response.strip()
        if s.startswith("{"):
            try:
                return _extract_text_from_completion(json.loads(s))
            except json.JSONDecodeError:
                pass
        return s

    if isinstance(response, dict):
        err = response.get("error")
        if isinstance(err, dict):
            em = err.get("message") or str(err)
            raise RuntimeError(f"API error: {em}")
        choices = response.get("choices") or []
        if choices:
            return _extract_text_from_choice(choices[0])
        for key in ("output_text", "text", "content", "message"):
            v = response.get(key)
            if isinstance(v, str) and v.strip():
                return v
        raise RuntimeError(f"Unexpected API response (no choices): {response!r}")

    choices = getattr(response, "choices", None)
    if choices:
        return _extract_text_from_choice(choices[0])
    return ""


def _ensure_images_dict(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, str):
        s = resp.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            f"Images API returned non-JSON string (prefix): {s[:300]!r}..."
        )

    if isinstance(resp, dict):
        err = resp.get("error")
        if isinstance(err, dict):
            em = err.get("message") or str(err)
            raise RuntimeError(f"API error: {em}")
        return resp

    model_dump = getattr(resp, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except TypeError:
            pass
    data = getattr(resp, "data", None)
    if data is not None:
        rows: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                rows.append(item)
            elif hasattr(item, "model_dump"):
                rows.append(item.model_dump())
            else:
                rows.append(
                    {
                        "b64_json": getattr(item, "b64_json", None),
                        "url": getattr(item, "url", None),
                        "revised_prompt": getattr(item, "revised_prompt", None),
                    }
                )
        return {"data": rows}

    raise RuntimeError(f"Unexpected images API response type: {type(resp)!r}")


def _first_image_fields(d: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    data = d.get("data") or []
    if not data:
        raise RuntimeError(f"No image entry in response: {d!r}")
    item = data[0]
    if isinstance(item, dict):
        b64 = item.get("b64_json")
        url = item.get("url")
        rp = item.get("revised_prompt") or ""
        if isinstance(rp, str):
            return b64, url, rp
        return b64, url, ""
    b64 = getattr(item, "b64_json", None)
    url = getattr(item, "url", None)
    rp = getattr(item, "revised_prompt", None) or ""
    return b64, url, rp if isinstance(rp, str) else ""


def _shorten_for_curl_preview(obj: Any, max_uri: int = 120) -> Any:
    """缩短 data:image...base64，避免 request 输出过长。"""
    if isinstance(obj, str):
        if obj.startswith("data:image") and len(obj) > max_uri:
            return (
                obj[:max_uri]
                + f"...<省略 {len(obj) - max_uri} 字符，完整 payload 见 SDK 请求>"
            )
        return obj
    if isinstance(obj, dict):
        return {k: _shorten_for_curl_preview(v, max_uri) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shorten_for_curl_preview(x, max_uri) for x in obj]
    return obj


def _curl_json_post(base_url: str, path: str, payload: Dict[str, Any]) -> str:
    """生成 Bash 下可用的 curl 预览（-d 使用 shlex.quote）。"""
    root = base_url.rstrip("/")
    p = path.strip("/")
    full = f"{root}/{p}"
    safe = _shorten_for_curl_preview(payload)
    body = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "curl -sS \\",
        f"  {shlex.quote(full)} \\",
        '  -H "Authorization: Bearer <API_KEY>" \\',
        '  -H "Content-Type: application/json" \\',
        f"  -d {shlex.quote(body)}",
    ]
    return "\n".join(lines)

def _curl_images_edit_hint(
    base_url: str, model: str, prompt: str, size: str, num_input_images: int = 1
) -> str:
    """multipart 预览；多图时与官方示例一致使用多个 image[] 字段。"""
    root = base_url.rstrip("/")
    full = f"{root}/images/edits"
    pr = shlex.quote(prompt or "")
    n = max(1, int(num_input_images))
    lines: List[str] = [
        "# images/edits（multipart）；GPT 图模支持多图输入，SDK 会把多张图作为 image 序列上传：",
        "curl -sS \\",
        f"  {shlex.quote(full)} \\",
        '  -H "Authorization: Bearer <API_KEY>" \\',
        f'  -F "model={model}" \\',
        f"  -F prompt={pr} \\",
        '  -F "n=1" \\',
        f'  -F "size={size}" \\',
    ]
    for i in range(n):
        if i < n - 1:
            lines.append(f'  -F "image[]=@image{i + 1}.png" \\')
        else:
            lines.append(f'  -F "image[]=@image{i + 1}.png"')
    return "\n".join(lines)

def _decode_image_tensor(b64: Optional[str], url: Optional[str]) -> torch.Tensor:
    if b64:
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    elif url:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-Xin-Openai/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    else:
        raise RuntimeError("Image response has neither b64_json nor url.")

    arr = np.asarray(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t.unsqueeze(0)


class OpenAPIImage:
    """
    OpenAI 兼容节点。
    - api_mode=`images`：官方 Images API（与其它兼容 `/v1/images/*` 的网关）
    - api_mode=`chat_completions`：官方 Chat Completions（多模态图+文，与其它兼容 `/v1/chat/completions` 的网关）
    参考图使用可选槽 image1～image10（最多 10 张）；chat 模式多张图依次写入消息；
    images 模式走 images.edit 时按顺序上传全部参考图（SDK：image 可为多文件序列）。
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "一只卡通海獭，扁平插画，白底",
                    },
                ),
                "api_mode": (
                    ["images", "chat_completions"],
                    {"default": "images"},
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://uclaude.cc/v1",
                        "placeholder": "请输入模型服务商的API地址",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "请输入模型服务商的API密钥",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gpt-image-2",
                        "placeholder": "请输入模型服务商的模型名称,例如：gpt-image-2",
                    },
                ),
                "size": (
                    [
                        "自动（auto）",
                        "1024x1024 - 1K（1:1方图）",
                        "1536x1024 - 1.5K（3:2横图）",
                        "1024x1536 - 1.5K（2:3竖图）",
                        "2048x2048 - 2K（1:1方图）",
                        "2880x2880 - 3K（1:1方图）",
                        "3840x2160 - 4K（16:9横图）",
                        "2160x3840 - 4K（9:16竖图）",
                    ],
                    {"default": "自动（auto）"},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "quality": (
                    [
                        "自动（auto）",
                        "低（low）",
                        "中（medium）",
                        "高（high）",
                    ],
                    {"default": "自动（auto）"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "text", "request", "response")
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "run"
    CATEGORY = "ComfyUI-Xin-Openai"

    def run(
        self,
        prompt: str,
        api_mode: str,
        base_url: str,
        api_key: str,
        model: str,
        size: str,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        image4: Optional[torch.Tensor] = None,
        image5: Optional[torch.Tensor] = None,
        image6: Optional[torch.Tensor] = None,
        image7: Optional[torch.Tensor] = None,
        image8: Optional[torch.Tensor] = None,
        image9: Optional[torch.Tensor] = None,
        image10: Optional[torch.Tensor] = None,
        quality: str = "自动（auto）",
    ):
        ref_images = _collect_optional_images(
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        )
        key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY") or os.getenv(
            "ARK_API_KEY"
        )
        if not key:
            return _failed_run_outputs(
                RuntimeError(
                    "缺少 API Key：填写 api_key 或设置环境变量 OPENAI_API_KEY / ARK_API_KEY。"
                ),
                "# 未发送请求：缺少 API Key。",
            )

        url = _normalize_base_url((base_url or "").strip())
        if not url:
            return _failed_run_outputs(
                RuntimeError("base_url 不能为空。"),
                "# 未发送请求：base_url 为空。",
            )

        mid = (model or "").strip()
        if not mid:
            return _failed_run_outputs(
                RuntimeError("model 不能为空。"),
                "# 未发送请求：model 为空。",
            )

        client = OpenAI(base_url=url, api_key=key)

        try:
            if api_mode == "chat_completions":
                return self._run_chat(client, url, prompt, mid, ref_images)

            return self._run_images(
                client, url, prompt, mid, size, ref_images, quality
            )
        except Exception as e:
            return _failed_run_outputs(e, "")

    def _run_chat(
        self,
        client: OpenAI,
        base_url: str,
        prompt: str,
        model_id: str,
        ref_images: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, str, str]:
        ml = (model_id or "").strip().lower()
        if "gpt-image" in ml:
            return _failed_run_outputs(
                RuntimeError(
                    "gpt-image-* 只能用于 api_mode=images（images.generate），不能用于 chat_completions。"
                ),
                "# chat/completions 未发送：不应使用 gpt-image-* 模型。",
            )

        has_img = len(ref_images) > 0
        content: List[Dict[str, Any]] = []
        for img_tensor in ref_images:
            data_uri = _tensor_to_image_data_uri(img_tensor)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        content.append({"type": "text", "text": prompt or ""})
        if not has_img and not (prompt or "").strip():
            return _failed_run_outputs(
                RuntimeError(
                    "chat 模式需要填写 prompt，或连接 image1～image10 至少一张图。"
                ),
                "# chat/completions 未发送：缺少文本 prompt 与参考图。",
            )

        payload = {"model": model_id, "messages": [{"role": "user", "content": content}]}
        req_preview = _curl_json_post(base_url, "chat/completions", payload)

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
            )
            raw = _response_to_raw_string(resp)
            text = _extract_text_from_completion(resp)
            out_img = _passthrough_or_blank(ref_images[0] if has_img else None)
            return (out_img, text, req_preview, raw)
        except Exception as e:
            return _failed_run_outputs(e, req_preview)

    def _run_images(
        self,
        client: OpenAI,
        base_url: str,
        prompt: str,
        mid: str,
        size: str,
        ref_images: List[torch.Tensor],
        quality: str,
    ) -> Tuple[torch.Tensor, str, str, str]:
        use_edit = len(ref_images) > 0

        api_size = _resolve_images_api_size(size)
        gen_kw: Dict[str, Any] = {
            "model": mid,
            "prompt": prompt or "",
            "n": 1,
            "size": api_size,
        }
        q = (quality or "").strip()
        api_q = _resolve_images_api_quality(q)
        if api_q is not None:
            gen_kw["quality"] = api_q

        if use_edit:
            req_preview = _curl_images_edit_hint(
                base_url, mid, prompt or "", api_size, len(ref_images)
            )
            edit_blobs: List[io.BytesIO] = []
            for idx, t in enumerate(ref_images):
                png = _tensor_to_png_bytes(t)
                bio = io.BytesIO(png)
                bio.name = f"image{idx + 1}.png"
                edit_blobs.append(bio)
            edit_image_arg: Any = (
                edit_blobs[0] if len(edit_blobs) == 1 else edit_blobs
            )
            edit_kw: Dict[str, Any] = {
                "model": mid,
                "image": edit_image_arg,
                "prompt": prompt or "",
                "n": 1,
                "size": api_size,
            }
            try:
                resp = _call_with_transient_retry(
                    lambda: client.images.edit(**edit_kw)
                )
                raw = _response_to_raw_string(resp)
                d = _ensure_images_dict(resp)
                b64, img_url, revised = _first_image_fields(d)
                out_image = _decode_image_tensor(b64, img_url)
                text_out = revised if revised else ""
                return (out_image, text_out, req_preview, raw)
            except Exception as e:
                return _failed_run_outputs(e, req_preview)
        else:
            gen_try = dict(gen_kw)
            gen_try["response_format"] = "b64_json"
            body_try = _shorten_for_curl_preview(gen_try)
            body_fallback = _shorten_for_curl_preview(dict(gen_kw))
            req_preview = "\n\n".join(
                [
                    "# ① 与 SDK 一致：首次尝试带 response_format=b64_json",
                    _curl_json_post(base_url, "images/generations", body_try),
                    "# ② 若失败则第二次请求（无 response_format），与 test.py 相同",
                    _curl_json_post(base_url, "images/generations", body_fallback),
                ]
            )
            gen_kw_try = dict(gen_kw)
            gen_kw_try["response_format"] = "b64_json"
            try:
                try:
                    resp = _call_with_transient_retry(
                        lambda: client.images.generate(**gen_kw_try)
                    )
                except Exception:
                    resp = _call_with_transient_retry(
                        lambda: client.images.generate(**gen_kw)
                    )
                raw = _response_to_raw_string(resp)
                d = _ensure_images_dict(resp)
                b64, img_url, revised = _first_image_fields(d)
                out_image = _decode_image_tensor(b64, img_url)
                text_out = revised if revised else ""
                return (out_image, text_out, req_preview, raw)
            except Exception as e:
                return _failed_run_outputs(e, req_preview)


NODE_CLASS_MAPPINGS = {
    "OpenAPIImage": OpenAPIImage,
    "OpenAPIVisionChat": OpenAPIImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAPIImage": "Xin OpenAI - Image/Chat",
    "OpenAPIVisionChat": "Xin OpenAI - Image/Chat",
}
