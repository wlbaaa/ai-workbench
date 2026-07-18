# 多模态 AI 大模型

整合 **图像识别 / AI绘图 / 图生视频 / 语音合成 / 文本对话** 五大能力的统一模型框架。

## 架构

```
MultimodalAI
├── Vision   (BLIP)          图像描述、分类
├── Drawing  (StableDiffusion)  文生图
├── Video    (AnimateDiff)   图生视频
├── Speech   (Bark)          文字转语音
└── Text     (Qwen2-0.5B)    大语言模型对话
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行演示（默认测 vision + text）
python demo.py

# 3. 指定模块
DEMO_MODULES=vision,drawing,speech python demo.py
```

## 使用示例

### Python API

```python
from model import MultimodalAI

ai = MultimodalAI()

# 图像识别
ai.load("vision")
caption = ai.describe_image("photo.jpg")
print(caption)

# AI 绘图
ai.load("drawing")
image = ai.text_to_image("一只太空猫", seed=42)
image.save("cat.png")

# 语音合成
ai.load("speech")
result = ai.text_to_speech("你好世界", output_path="hello.wav")

# 文本对话
ai.load("text")
reply = ai.chat([
    {"role": "user", "content": "介绍一下量子计算"}
])
print(reply)
```

### HTTP API 服务

```bash
python server.py --port 8080
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/vision/describe` | POST | 图像描述 |
| `/vision/classify` | POST | 图像分类 |
| `/drawing/generate` | POST | 文生图 |
| `/video/generate` | POST | 图生视频 |
| `/speech/generate` | POST | 文字转语音 |
| `/chat` | POST | 文本对话 |

### API 调用示例

```bash
# 图像识别
curl -X POST http://localhost:8080/vision/describe \
  -F "file=@photo.jpg"

# 文生图
curl -X POST http://localhost:8080/drawing/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"赛博朋克城市", "seed": 42}' \
  -o output.png

# 语音合成
curl -X POST http://localhost:8080/speech/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，世界"}' \
  -o output.wav
```

## 硬件要求

| 模块 | GPU | CPU |
|------|-----|-----|
| Vision | 2 GB VRAM | 可用（较慢） |
| Drawing | 6 GB VRAM | 不推荐 |
| Video | 8 GB VRAM | 不支持 |
| Speech | 4 GB VRAM | 可用 |
| Text | 2 GB VRAM | 可用 |

## 模型来源

所有模型从 HuggingFace 自动下载，首次运行需联网。

- BLIP: `Salesforce/blip-image-captioning-base`
- Stable Diffusion: `runwayml/stable-diffusion-v1-5`
- AnimateDiff: `guoyww/animatediff-motion-adapter-v1-5-2`
- Bark: `suno/bark-small`
- Qwen2: `Qwen/Qwen2-0.5B-Instruct`
