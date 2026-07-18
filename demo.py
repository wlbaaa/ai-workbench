"""
多模态 AI 大模型 - 演示脚本
运行: python demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalAI
from pathlib import Path


def demo_vision(ai: MultimodalAI):
    print("\n" + "=" * 50)
    print("  模块 1: 图像识别")
    print("=" * 50)
    print("加载中...")
    ai.load("vision")

    # 使用一张测试图片（如果提供了就使用，否则提示）
    test_img = os.environ.get("TEST_IMAGE", "")
    if test_img and Path(test_img).exists():
        caption = ai.describe_image(test_img)
        print(f"\n  图像描述: {caption}")

        results = ai.classify_image(test_img)
        print(f"\n  分类结果 (Top-5):")
        for r in results:
            print(f"    {r['label']:30s} {r['score']:.3f}")
    else:
        print("\n  设置 TEST_IMAGE 环境变量指向一张图片即可测试")
        print('  例: TEST_IMAGE=./photo.jpg python demo.py')


def demo_drawing(ai: MultimodalAI):
    print("\n" + "=" * 50)
    print("  模块 2: AI 绘图 (文生图)")
    print("=" * 50)
    prompt = os.environ.get("DRAW_PROMPT", "一只在星空下弹吉他的猫，赛博朋克风格")
    print(f"\n  Prompt: {prompt}")
    print("  生成中... (需 GPU，CPU 较慢)")
    ai.load("drawing")
    image = ai.text_to_image(prompt, seed=42)
    output = "output_drawing.png"
    image.save(output)
    print(f"  已保存: {output}")


def demo_video(ai: MultimodalAI):
    print("\n" + "=" * 50)
    print("  模块 3: 图生视频")
    print("=" * 50)
    prompt = os.environ.get("VIDEO_PROMPT", "a beautiful sunset over mountains, smooth camera pan")
    print(f"\n  Prompt: {prompt}")
    print("  生成中... (需 GPU)")
    ai.load("video")
    frames = ai.image_to_video(prompt, num_frames=16, seed=42)
    output = "output_video.gif"
    ai.save_video_gif(frames, output, fps=8)
    print(f"  已保存: {output}")


def demo_speech(ai: MultimodalAI):
    print("\n" + "=" * 50)
    print("  模块 4: 语音合成")
    print("=" * 50)
    text = os.environ.get("TTS_TEXT", "你好！我是多模态人工智能模型，很高兴为你服务。")
    print(f"\n  文本: {text}")
    ai.load("speech")
    result = ai.text_to_speech(text, voice_preset="zh_speaker_1", output_path="output_speech.wav")
    print(f"  已保存: {result['output_path']}")
    print(f"  采样率: {result['sample_rate']} Hz")
    print(f"  时长: {len(result['audio']) / result['sample_rate']:.1f} 秒")


def demo_chat(ai: MultimodalAI):
    print("\n" + "=" * 50)
    print("  模块 5: 文本对话")
    print("=" * 50)
    ai.load("text")

    messages = [
        {"role": "system", "content": "你是一个专业的多模态AI助手，用中文回答。"},
        {"role": "user", "content": "请用一句话介绍多模态AI"},
    ]
    print(f"\n  用户: {messages[-1]['content']}")
    print("  思考中...")
    reply = ai.chat(messages, max_new_tokens=128)
    print(f"  AI: {reply}")


def main():
    print("=" * 50)
    print("  多模态 AI 大模型 - 功能演示")
    print("=" * 50)

    ai = MultimodalAI()
    print(f"\n  设备: {ai.info()['device'].upper()}")

    # 按需运行 - 用环境变量控制
    demos = {
        "vision": demo_vision,
        "drawing": demo_drawing,
        "video": demo_video,
        "speech": demo_speech,
        "text": demo_chat,
    }

    run = os.environ.get("DEMO_MODULES", "vision,text").split(",")
    run = [m.strip() for m in run if m.strip() in demos]

    for module in run:
        try:
            demos[module](ai)
        except Exception as e:
            print(f"\n  [{module}] 错误: {e}")

    print("\n" + "=" * 50)
    print("  演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
