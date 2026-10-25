"""Local runner for Phased Consistency Model (PCM) inference.

Compatible with CPU and CUDA devices.
"""

import argparse
import sys
import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import TCDScheduler


def generate(prompt: str, num_inference_steps: int = 2, output_path: str = "output_pcm.png") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[*] Inference device: {device.upper()} (dtype={dtype})")
    print(f"[*] Base model: runwayml/stable-diffusion-v1-5")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(device)

    lora_weight = f"pcm_sd15_smallcfg_{num_inference_steps}step_converted.safetensors"
    print(f"[*] Loading PCM LoRA weight ({num_inference_steps}-step): {lora_weight}")
    pipe.load_lora_weights(
        "wangfuyun/PCM_Weights",
        weight_name=lora_weight,
        subfolder="sd15",
    )

    pipe.scheduler = TCDScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        timestep_spacing="trailing",
    )

    print(f"[*] Generating image for prompt: '{prompt}' ({num_inference_steps} steps)...")
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
    )
    image = result.images[0]
    image.save(output_path)
    print(f"[✓] Done! Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phased Consistency Model inference locally")
    parser.add_argument("--prompt", type=str, default="A sleek cyberpunk sports car parked in rain at neon night, 8k", help="Text prompt")
    parser.add_argument("--steps", type=int, default=2, choices=[2, 4, 8, 16], help="Inference steps (default: 2)")
    parser.add_argument("--output", type=str, default="pcm_sample.png", help="Output file path")
    args = parser.parse_args()

    generate(args.prompt, args.steps, args.output)
