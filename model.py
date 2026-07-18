"""
多模态 AI 大模型 - 核心引擎
整合图像识别、文生图、图生视频、语音合成、文本对话
"""

import os
import warnings
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

warnings.filterwarnings("ignore")

# ============================================================
# 设备检测
# ============================================================
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"[MultimodalAI] 设备: {DEVICE.upper()}")


class MultimodalAI:
    """
    多模态 AI 大模型

    功能模块:
      - vision:    图像识别 / 描述 / 分类
      - drawing:   文生图 (Stable Diffusion)
      - video:     图生视频 (AnimateDiff)
      - speech:    文字转语音 (Bark)
      - text:      大语言模型对话 (Qwen2)

    用法:
      ai = MultimodalAI()
      ai.load("vision")               # 按需加载模块
      caption = ai.describe_image("photo.jpg")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "multimodal-ai"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}

    # ============================================================
    # 模块加载
    # ============================================================
    def load(self, module: str):
        """按需加载指定模块: vision / drawing / video / speech / text / all"""
        modules = {
            "vision": self._load_vision,
            "drawing": self._load_drawing,
            "video": self._load_video,
            "speech": self._load_speech,
            "text": self._load_text,
        }
        if module == "all":
            for key in modules:
                try: modules[key]()
                except Exception as e: print(f"  [跳过] {key}: {e}")
        elif module in modules:
            modules[module]()
        else:
            raise ValueError(f"未知模块: {module}. 可选: {list(modules.keys())}")

    def _load_vision(self):
        """加载图像识别模型 (BLIP)"""
        if "vision" in self._models: return
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("[Vision] 加载 BLIP 图像描述模型...")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
        ).to(DEVICE)
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=self.cache_dir,
        )
        self._models["vision"] = model
        self._processors["vision"] = processor
        print("[Vision] 就绪")

    def _load_drawing(self):
        """加载文生图模型 (Stable Diffusion)"""
        if "drawing" in self._models: return
        from diffusers import StableDiffusionPipeline
        print("[Drawing] 加载 Stable Diffusion...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
            safety_checker=None,
        )
        if DEVICE == "cuda":
            pipe = pipe.to("cuda")
            try: pipe.enable_xformers_memory_efficient_attention()
            except: pass
        self._models["drawing"] = pipe
        print("[Drawing] 就绪")

    def _load_video(self):
        """加载图生视频模型 (AnimateDiff)"""
        if "video" in self._models: return
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        from diffusers.utils import export_to_gif
        print("[Video] 加载 AnimateDiff...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            "emilianJR/epiCRealism",
            motion_adapter=adapter,
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="trailing",
            steps_offset=1,
        )
        if DEVICE == "cuda":
            pipe = pipe.to("cuda")
            try: pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
            except: pass
        self._models["video"] = pipe
        print("[Video] 就绪")

    def _load_speech(self):
        """加载语音合成模型 (Bark)"""
        if "speech" in self._models: return
        from transformers import AutoProcessor, BarkModel
        print("[Speech] 加载 Bark 语音模型...")
        model = BarkModel.from_pretrained(
            "suno/bark-small",
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            "suno/bark-small",
            cache_dir=self.cache_dir,
        )
        self._models["speech"] = model
        self._processors["speech"] = processor
        print("[Speech] 就绪")

    def _load_text(self):
        """加载文本对话模型 (Qwen2)"""
        if "text" in self._models: return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[Text] 加载 Qwen2 对话模型...")
        model_id = "Qwen/Qwen2-0.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
        self._models["text"] = model
        self._processors["text"] = tokenizer
        print("[Text] 就绪")

    # ============================================================
    # 推理接口
    # ============================================================

    # --- 图像识别 ---
    def describe_image(
        self, image: Union[str, Image.Image, np.ndarray], max_length: int = 60
    ) -> str:
        """输入图片路径/PIL图像/ndarray，返回中文描述"""
        self._ensure_loaded("vision")
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        processor = self._processors["vision"]
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        out = self._models["vision"].generate(**inputs, max_length=max_length)
        return processor.decode(out[0], skip_special_tokens=True)

    def classify_image(self, image: Union[str, Image.Image]) -> List[Dict[str, float]]:
        """图像分类，返回 top-5 类别及置信度"""
        self._ensure_loaded("vision")
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        from transformers import pipeline
        classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=0 if DEVICE == "cuda" else -1,
        )
        return classifier(image, top_k=5)

    # --- 文生图 ---
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """输入文字描述，生成图像 (PIL Image)"""
        self._ensure_loaded("drawing")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        result = self._models["drawing"](
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality, distorted",
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

    # --- 图生视频 ---
    def image_to_video(
        self,
        prompt: str,
        num_frames: int = 16,
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> List[Image.Image]:
        """输入描述，生成短视频帧序列"""
        self._ensure_loaded("video")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        result = self._models["video"](
            prompt=prompt,
            negative_prompt="blurry, distorted, low quality",
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.frames[0]

    def save_video_gif(self, frames: List[Image.Image], output_path: str, fps: int = 8):
        """将帧序列保存为 GIF"""
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        return output_path

    # --- 语音合成 ---
    def text_to_speech(
        self,
        text: str,
        voice_preset: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        文字转语音
        voice_preset: 音色预设
          - "zh_speaker_0" ~ "zh_speaker_9"  中文音色
          - "v2/zh_speaker_1"                v2 中文
        """
        self._ensure_loaded("speech")
        processor = self._processors["speech"]
        model = self._models["speech"]

        if voice_preset:
            voice_preset = f"[{voice_preset}]"
        else:
            voice_preset = ""

        inputs = processor(voice_preset + text, return_tensors="pt").to(DEVICE)
        audio_array = model.generate(**inputs, do_sample=True)
        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = model.generation_config.sample_rate

        if output_path:
            import soundfile as sf
            sf.write(output_path, audio_array, sample_rate)

        return {
            "audio": audio_array,
            "sample_rate": sample_rate,
            "output_path": output_path,
        }

    # --- 文本对话 ---
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        多轮对话
        messages = [
            {"role": "system", "content": "你是AI助手"},
            {"role": "user", "content": "你好"},
        ]
        """
        self._ensure_loaded("text")
        tokenizer = self._processors["text"]
        model = self._models["text"]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(response, skip_special_tokens=True).strip()

    # ============================================================
    # 工具方法
    # ============================================================
    def _ensure_loaded(self, module: str):
        if module not in self._models:
            self.load(module)

    @staticmethod
    def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def info(self) -> Dict[str, bool]:
        """查看各模块加载状态"""
        return {
            "device": DEVICE,
            "vision": "vision" in self._models,
            "drawing": "drawing" in self._models,
            "video": "video" in self._models,
            "speech": "speech" in self._models,
            "text": "text" in self._models,
        }
