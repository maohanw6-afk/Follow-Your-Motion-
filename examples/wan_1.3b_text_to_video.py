import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download


# Download models
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)


# Video-to-video
video = VideoData("video1.mp4", height=480, width=832)
video = pipe(
    prompt="Documentary photography style, a lively puppy wearing black sunglasses running rapidly across a lush green grassy field. The puppy has tan fur, is wearing black sunglasses, has its two ears standing erect, with a focused and joyful expression. Sunlight shines on its body, making its fur look exceptionally soft and glossy. The background is an expansive grassy field, dotted occasionally with a few wildflowers, in the distance the blue sky and a few white clouds are faintly visible. The perspective is distinct, capturing the dynamic energy of the puppy running and the vitality of the surrounding grassland. Side profile medium tracking shot perspective.",
    negative_prompt="Gaudy colors, overexposed, static, blurred details, subtitles, style, artwork, painting, image, still, overall gray, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, morbidly deformed limbs, fused fingers, motionless frame, cluttered background, three legs, crowded background, walking backward.",
    input_video=video, denoising_strength=0.7,
    num_inference_steps=50,
    seed=1, tiled=True
)
save_video(video, "video2.mp4", fps=15, quality=5)
