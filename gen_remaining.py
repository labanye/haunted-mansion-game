#!/usr/bin/env python3
"""Generate all remaining atmospheric images using cached SD 1.5 model."""
import torch, gc, os, sys
from diffusers import StableDiffusionPipeline

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.chdir(r'C:\Users\Administrator\Documents\Codex\2026-07-26\plugin-computer-use-openai-bundled\repo')

print('Loading model from cache...')
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
print('Model loaded on CUDA!')

IMAGES = {
    'piano': 'dark victorian living room grand piano in corner heavy burgundy curtains closed single candle flame dim light gothic horror atmosphere dust particles',
    'ritual': 'dark stone basement floor large occult ritual circle made of chalk inverted triangle inside circle three black candles casting shadows gothic horror',
    'attic': 'dusty victorian attic moonlight streaming through small window old furniture covered sheets cobwebs wooden floorboards horror atmosphere abandoned',
    'letter': 'close up antique letter on dark wooden table red wax seal with crest yellowed parchment paper victorian era fountain pen ink candlelight macro',
    'diary': 'open leather diary on victorian mahogany desk handwritten pages sepia ink vintage candlelight dark atmosphere gothic brass candlestick',
}

for name, prompt in IMAGES.items():
    out = f'assets/{name}.png'
    if os.path.exists(out):
        print(f'SKIP {name} - already exists')
        continue
    print(f'Generating {name}...')
    try:
        img = pipe(
            prompt,
            negative_prompt='bright, sunny, daytime, happy, cartoon, oversaturated, modern',
            num_inference_steps=25,
            width=512, height=512,
            guidance_scale=7.5
        ).images[0]
        img.save(out)
        size = os.path.getsize(out) // 1024
        print(f'  -> {out} ({size} KB)')
    except Exception as e:
        print(f'  ERROR: {e}')
    gc.collect()
    torch.cuda.empty_cache()

print('\nAll done!')
