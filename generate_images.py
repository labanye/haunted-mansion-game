#!/usr/bin/env python3
"""Generate atmospheric images for 霍桑庄园 game using Stable Diffusion."""
import os, sys, gc, json, base64
from pathlib import Path

# Ensure we use the right venv
os.environ['PATH'] = r'C:\Users\Administrator\Documents\Codex\2026-07-26\plugin-computer-use-openai-bundled\repo\venv_img\Scripts' + os.pathsep + os.environ['PATH']

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
except ImportError as e:
    print(f"ERROR: {e}")
    print("Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print("Then: pip install diffusers transformers accelerate")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent / 'assets'
OUTPUT_DIR.mkdir(exist_ok=True)

# Prompts for each image
IMAGES = {
    "mansion": {
        "prompt": "dark gothic victorian mansion exterior at night, heavy rain, fog, two tall windows with warm dim light, iron gate, horror atmosphere, cinematic, highly detailed, dark moody",
        "negative": "bright, sunny, daytime, happy, cartoon, anime, oversaturated",
        "file": "mansion.png"
    },
    "letter": {
        "prompt": "close up of an antique letter on dark wooden table, red wax seal with crest, yellowed parchment paper, Victorian era, candlelight, fountain pen ink, dark moody, macro photography",
        "negative": "bright, modern, digital, clean, daytime",
        "file": "letter.png"
    },
    "diary": {
        "prompt": "open leather diary on Victorian mahogany desk, handwritten pages in sepia ink, vintage, candlelight, dark atmosphere, gothic, antique brass candlestick nearby, shallow depth of field",
        "negative": "bright, modern, digital, clean, daytime",
        "file": "diary.png"
    },
    "ritual": {
        "prompt": "dark stone basement floor with large occult ritual circle drawn in chalk, inverted triangle inside circle, black candles at three points, torchlight casting shadows, gothic horror, dungeons, dark occult",
        "negative": "bright, cheerful, colorful, daytime, modern",
        "file": "ritual.png"
    },
    "piano": {
        "prompt": "dark Victorian living room with grand piano in corner, heavy burgundy curtains closed, single candle on piano, dust particles in light beam, gothic atmosphere, old mansion interior, horror",
        "negative": "bright, sunny, modern, clean, happy",
        "file": "piano.png"
    },
    "attic": {
        "prompt": "dusty victorian attic with old furniture covered in white sheets, moonlight streaming through small window, cobwebs, wooden floorboards, boxes and trunks, horror atmosphere, abandoned",
        "negative": "bright, clean, modern, daytime, cheerful",
        "file": "attic.png"
    }
}

def generate():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    
    # Load model
    print("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_attention_slicing()
    
    if torch.cuda.is_available():
        pipe.enable_sequential_cpu_offload()
    
    results = {}
    
    for key, cfg in IMAGES.items():
        out_path = OUTPUT_DIR / cfg["file"]
        if out_path.exists():
            print(f"[SKIP] {cfg['file']} already exists")
            continue
        
        print(f"[GEN] {key}: {cfg['prompt'][:50]}...")
        try:
            image = pipe(
                cfg["prompt"],
                negative_prompt=cfg.get("negative", ""),
                num_inference_steps=25,
                width=512,
                height=512,
                guidance_scale=7.5
            ).images[0]
            
            image.save(out_path)
            print(f"  -> saved {out_path} ({out_path.stat().st_size // 1024} KB)")
            
            # Convert to base64 for embedding
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            results[key] = b64
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        # Clear memory between generations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save base64 data for embedding
    with open(OUTPUT_DIR / "images_b64.json", "w") as f:
        json.dump(results, f)
    print(f"\nDone! Generated {len(results)}/{len(IMAGES)} images")
    print(f"Base64 data saved to {OUTPUT_DIR / 'images_b64.json'}")

if __name__ == "__main__":
    generate()
