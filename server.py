"""
多模态 AI 大模型 - HTTP API 服务
启动: python server.py --port 8080
"""

import sys
import os
import io
import json
import base64
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalAI

from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)
ai = MultimodalAI()


# ---------- 健康检查 ----------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "MultimodalAI", "status": "running", "info": ai.info()})


# ---------- 图像识别 ----------
@app.route("/vision/describe", methods=["POST"])
def vision_describe():
    """POST {image: base64} 或 multipart file"""
    image = _load_image(request)
    if image is None:
        return jsonify({"error": "请上传图片 (base64 或 multipart file)"}), 400
    caption = ai.describe_image(image)
    return jsonify({"caption": caption})


@app.route("/vision/classify", methods=["POST"])
def vision_classify():
    image = _load_image(request)
    if image is None:
        return jsonify({"error": "请上传图片"}), 400
    results = ai.classify_image(image)
    return jsonify({"results": results})


# ---------- AI 绘图 ----------
@app.route("/drawing/generate", methods=["POST"])
def drawing_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 prompt"}), 400
    seed = data.get("seed")
    image = ai.text_to_image(
        prompt=prompt,
        negative_prompt=data.get("negative_prompt", ""),
        num_steps=data.get("num_steps", 25),
        guidance_scale=data.get("guidance_scale", 7.5),
        seed=seed,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---------- 图生视频 ----------
@app.route("/video/generate", methods=["POST"])
def video_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "缺少 prompt"}), 400
    frames = ai.image_to_video(
        prompt=prompt,
        num_frames=data.get("num_frames", 16),
        seed=data.get("seed", 42),
    )
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=frames[1:],
        duration=125, loop=0,
    )
    buf.seek(0)
    return send_file(buf, mimetype="image/gif")


# ---------- 语音合成 ----------
@app.route("/speech/generate", methods=["POST"])
def speech_generate():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "缺少 text"}), 400
    result = ai.text_to_speech(
        text=text,
        voice_preset=data.get("voice_preset"),
    )
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, result["audio"], result["sample_rate"], format="WAV")
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


# ---------- 文本对话 ----------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "缺少 messages"}), 400
    reply = ai.chat(
        messages=messages,
        max_new_tokens=data.get("max_new_tokens", 512),
        temperature=data.get("temperature", 0.7),
        top_p=data.get("top_p", 0.9),
    )
    return jsonify({"reply": reply})


# ---------- 工具函数 ----------
def _load_image(req):
    """从请求中提取 PIL Image"""
    from PIL import Image
    # multipart file
    if "file" in req.files:
        f = req.files["file"]
        return Image.open(f.stream).convert("RGB")
    # base64 JSON
    data = req.get_json(silent=True) or {}
    b64 = data.get("image", "")
    if b64:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"MultimodalAI API 服务启动: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
