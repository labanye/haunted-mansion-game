#!/usr/bin/env python3
"""Generate ALL missing room + item images for Shadow Manor game."""
import torch, gc, os, sys
from diffusers import StableDiffusionPipeline

BASE = r'C:\Users\Administrator\Documents\Codex\2026-07-26\plugin-computer-use-openai-bundled\repo'
os.chdir(BASE)
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
ASSETS = os.path.join(BASE, 'assets')

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
print('Model on CUDA!')

def gen(name, prompt, neg='bright, sunny, daytime, cartoon, oversaturated, modern, text, letters'):
    out = os.path.join(ASSETS, f'{name}.png')
    if os.path.exists(out):
        print(f'SKIP {name}')
        return
    print(f'GEN {name}...')
    try:
        img = pipe(prompt, negative_prompt=neg,
                   num_inference_steps=25, width=512, height=512,
                   guidance_scale=7.5).images[0]
        img.save(out)
        print(f'  -> {os.path.getsize(out)//1024} KB')
    except Exception as e:
        print(f'  ERROR: {e}')
    gc.collect()
    torch.cuda.empty_cache()

# ===== ROOM IMAGES (6 missing) =====
gen('kitchen',
    'dark victorian kitchen cast iron stove black pot with brown residue cabinet doors with scratch marks from inside dim candlelight gothic horror abandoned mansion',
    'bright, clean, modern, happy, daytime, cartoon')

gen('dining',
    'victorian dining room long mahogany table set for three silver candlesticks tarnished overturned teacup broken in half dark red stains on tablecloth gothic horror',
    'bright, cheerful, modern, daytime, cartoon')

gen('study',
    'dark victorian study floor to ceiling bookshelves old desk with leather notebook VEN on cover anatomy diploma on wall specimen jars on windowsill gothic horror',
    'bright, modern, clean, daytime, cartoon')

gen('hallway',
    'narrow dark victorian hallway six portrait paintings on wall sixth portrait face scratched out candle wall sconce still burning moonlight十字 shadow on floor trapdoor moonlight gothic horror',
    'bright, modern, daytime, cartoon, cheerful')

gen('bedroom',
    'dark victorian master bedroom crimson bedding lump under covers like a body vanity mirror with red lipstick writing all over wardrobe slightly open gothic horror atmosphere',
    'bright, modern, daytime, cartoon, cheerful')

gen('bathroom',
    'dark victorian bathroom clawfoot bathtub half full of stagnant water small green doll with one eye at bottom mirror with red lipstick writing hidden niche behind shower curtain',
    'bright, modern, daytime, cartoon, cheerful')

# ===== KEY ITEM IMAGES (matching game text exactly) =====
gen('brass_key',
    'close up antique brass key with faded yellow paper tag on dark wood table victorian era macro photography shallow depth of field warm dim light',
    'bright, modern, colorful, cartoon, text, letters')

gen('sheet_music',
    'close up vintage sheet music partially burned blackened edges hole burned through center first and last notes circled in red ink victorian piano score on dark surface',
    'bright, modern, colorful, cartoon, text, letters visible')

gen('revolver',
    'close up antique six shot revolver on wooden floorboards dark rusty well maintained old west firearm victorian era dim candlelight macro photography',
    'bright, modern, colorful, cartoon, shiny')

gen('family_photo',
    'faded sepia victorian family photograph three people doctor father mother and seven year old daughter formal portrait in oval frame dusty dark wooden surface',
    'bright, colorful, modern, cartoon, happy')

gen('trophy',
    'gold victorian trophy cup anatomy competition first place engraving on dark wooden shelf dust cobwebs brass tarnished hollow base macro photography warm candlelight',
    'bright, modern, colorful, cartoon, shiny, text')

gen('ritual_key',
    'three metallic key fragments assembled together on dark stone surface circular pattern with triangle occult symbols ancient brass key complete intricate metalwork dim candlelight',
    'bright, modern, colorful, cartoon')

gen('lily_drawing',
    'childs crayon drawing on paper little girl lying on table three faceless figures standing around black and red colors dark disturbing victorian era childrens art',
    'bright, colorful, happy, cartoon, cheerful')

gen('phonograph',
    'antique phonograph with large brass horn cobwebs on horn victorian era on dark wooden table dim warm light dust particles vintage audio player',
    'bright, modern, colorful, cartoon, shiny')

gen('locket',
    'heart shaped gold locket on chain opened revealing miniature photograph mother holding baby girl engraved inside says mommy loves you forever victorian antique macro',
    'bright, modern, colorful, cartoon, text, letters')

gen('ven_notebook',
    'leather bound notebook open on desk handwritten latin words VENI VIDI VICI on pages victorian dark study brown leather cover vintage ink fountain pen nearby',
    'bright, modern, colorful, cartoon, text visible')

gen('iron_box',
    'small iron strongbox with combination lock on dark wooden shelf medical storage cabinet engraved words fathers hope victorian era rusted metal dim light',
    'bright, modern, colorful, cartoon, text, letters')

print('\n=== ALL GENERATION COMPLETE ===')
