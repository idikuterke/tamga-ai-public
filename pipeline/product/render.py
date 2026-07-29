import sys
import json
import random
import math
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageColor, ImageChops, ImageOps

PRODUCT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PRODUCT_DIR))

from rules_engine import SpellingEngine

SCHEMA_PATH = PRODUCT_DIR.parent.parent / "gokturk_labels_v1_locked.json"
FONTS_DIR = PRODUCT_DIR.parent / "fonts"

STYLES = {
    "plain":     {"bg": (255, 255, 255), "fg": (0, 0, 0),       "effects": [],                   "texture_path": "textures/plain.png",     "blend_mode": "normal"},
    "stone":     {"bg": (180, 178, 170), "fg": (40, 40, 40),    "effects": ["noise"],            "texture_path": "textures/stone.png",     "blend_mode": "multiply"},
    "gold":      {"bg": (60, 40, 20),    "fg": (255, 215, 0),   "effects": ["gradient"],         "texture_path": "textures/gold.png",      "blend_mode": "normal"},
    "neon":      {"bg": (10, 10, 30),    "fg": (0, 255, 200),   "effects": ["glow", "bloom"],    "texture_path": "textures/neon.png",      "blend_mode": "normal"},
    "wood":      {"bg": (110, 70, 40),   "fg": (220, 190, 100), "effects": ["grain"],            "texture_path": "textures/wood.png",      "blend_mode": "screen"},
    "paper":     {"bg": (245, 235, 210), "fg": (40, 30, 20),    "effects": ["vignette"],         "texture_path": "textures/paper.png",     "blend_mode": "multiply"},
    "leather":   {"bg": (80, 40, 20),    "fg": (40, 20, 10),    "effects": ["wear"],             "texture_path": "textures/leather.png",   "blend_mode": "multiply"},
    "parchment": {"bg": (230, 215, 170), "fg": (50, 30, 10),    "effects": ["stains", "burn_edges"], "texture_path": "textures/parchment.png", "blend_mode": "multiply"},
    "fircha":    {"bg": (240, 235, 225), "fg": (25, 25, 25),    "effects": ["brush", "ink_bleed"],   "texture_path": "textures/paper.png",     "blend_mode": "multiply"},
    "ink_bleed": {"bg": (235, 228, 215), "fg": (20, 15, 10),    "effects": ["ink_bleed_effect"],     "texture_path": "textures/paper.png",     "blend_mode": "multiply"},
    "stamp":     {"bg": (225, 215, 195), "fg": (139, 0, 0),     "effects": ["stamp_effect"],         "texture_path": "textures/parchment.png", "blend_mode": "multiply"},
    "carved":    {"bg": (160, 158, 150), "fg": (50, 50, 50),    "effects": ["carved_effect"],        "texture_path": "textures/stone.png",     "blend_mode": "multiply"},
    "chalk":     {"bg": (40, 40, 40),    "fg": (216, 212, 200), "effects": ["chalk_effect"],       "texture_path": "textures/stone.png",     "blend_mode": "normal"},
    "ember":     {"bg": (15, 10, 10),    "fg": (255, 180, 50),  "effects": ["ember_effect"],       "texture_path": "textures/plain.png",     "blend_mode": "normal"},
    "ash":       {"bg": (140, 138, 135), "fg": (138, 133, 128), "effects": ["ash_effect"],         "texture_path": "textures/stone.png",     "blend_mode": "normal"},
    "stencil":   {"bg": (200, 195, 185), "fg": (32, 32, 32),    "effects": ["stencil_effect"],     "texture_path": "textures/wood.png",      "blend_mode": "normal"},
}

STYLE_TO_TEXTURE_SUGGESTION = {
    "plain": ["plain.png", "stone.png"],
    "fircha": ["paper.png", "parchment.png"],
    "ink_bleed": ["paper.png", "parchment.png"],
    "stamp": ["parchment.png", "paper.png"],
    "parchment": ["parchment.png", "paper.png"],
    "carved": ["stone.png", "gold.png", "wood.png"],
    "chalk": ["stone.png", "plain.png"],
    "ember": ["plain.png", "neon.png", "stone.png"],
    "ash": ["stone.png", "plain.png", "wood.png"],
    "stencil": ["wood.png", "leather.png", "plain.png"],
    # Geriye dönük uyumluluk (Backward compatibility) eşlemeleri:
    "stone": ["stone.png"],
    "gold": ["gold.png"],
    "neon": ["neon.png"],
    "wood": ["wood.png"],
    "paper": ["paper.png"],
    "leather": ["leather.png"],
}

def get_texture_variations(style: str) -> list[Path]:
    tex_dir = PRODUCT_DIR / "textures"
    if not tex_dir.exists():
        return []
    style_lower = style.lower()
    if style_lower in ["fircha", "ink", "ink_bleed"]:
        style_lower = "paper"
    elif style_lower in ["stamp"]:
        style_lower = "parchment"
    elif style_lower in ["carved", "chalk", "ash"]:
        style_lower = "stone"
    elif style_lower in ["ember"]:
        style_lower = "plain"
    elif style_lower in ["stencil"]:
        style_lower = "wood"
    matches = []
    for f in tex_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            fname_lower = f.name.lower()
            if style_lower in fname_lower:
                matches.append(f)
    return sorted(matches)

get_texture_variants = get_texture_variations

# Lazy loading of schema and engine
_schema = None
_class_meta = None
_spelling_engine = None

def _get_engine():
    global _schema, _class_meta, _spelling_engine
    if _spelling_engine is None:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            _schema = json.load(f)
        _class_meta = {c["id"]: c for c in _schema["classes"]}
        _spelling_engine = SpellingEngine(str(SCHEMA_PATH))
    return _spelling_engine, _class_meta

STYLE_DECISION_TABLE = {
    "plain": {
        "font": "Gokturk-Regular.ttf",
        "effect": None,
        "fallback": "Gokturk-Regular.ttf",
    },
    "fircha": {
        "font": "Gokturk-Oblique.ttf",
        "effect": "brush",
        "fallback": "Gokturk-Regular.ttf",
    },
    "ink_bleed": {
        "font": "Gokturk-Oblique.ttf",
        "effect": "ink_bleed",
        "fallback": "Gokturk-Regular.ttf",
    },
    "chalk": {
        "font": "Gokturk-Light.ttf",
        "effect": "grain",
        "fallback": "Gokturk-Regular.ttf",
    },
    "ash": {
        "font": "Gokturk-Light.ttf",
        "effect": "fade",
        "fallback": "Gokturk-Light.ttf",
    },
    "stamp": {
        "font": "Gokturk-Bold.ttf",
        "effect": "stamp",
        "fallback": "Gokturk-Bold.ttf",
    },
    "carved": {
        "font": "Gokturk-Bold.ttf",
        "effect": "bevel",
        "fallback": "Gokturk-Bold.ttf",
    },
    "stencil": {
        "font": "Gokturk-Condensed.ttf",
        "effect": "stencil",
        "fallback": "Gokturk-Regular.ttf",
    },
    "parchment": {
        "font": "Gokturk-Regular.ttf",
        "effect": "paper",
        "fallback": "Gokturk-Regular.ttf",
    },
    "ember": {
        "font": "Gokturk-Bold.ttf",
        "effect": "glow",
        "fallback": "Gokturk-Bold.ttf",
    },
    "neon": {
        "font": "Gokturk-Bold.ttf",
        "effect": "cyan_glow",
        "fallback": "Gokturk-Bold.ttf",
    },
}

# Geriye dönük uyumluluk takma adı
STYLE_TO_FONT_MAPPING = {k: v["font"] for k, v in STYLE_DECISION_TABLE.items()}

def resolve_style(style_name: str) -> dict:
    """3 katmanlı stil karar tablosu çözücüsü: style -> {font, effect, fallback}."""
    spec = STYLE_DECISION_TABLE.get(style_name, STYLE_DECISION_TABLE["plain"])
    return {
        "font": spec["font"],
        "effect": spec["effect"],
        "fallback": spec["fallback"],
    }

def get_font_path(font_name: str, fallback_font: str = "Gokturk-Regular.ttf") -> Path:
    fp = FONTS_DIR / font_name
    if fp.exists():
        return fp
    fb_p = FONTS_DIR / fallback_font
    if fb_p.exists():
        return fb_p
    fp = FONTS_DIR / "NotoSansOldTurkic-Regular.ttf"
    if fp.exists():
        return fp
    # fallback to any ttf
    for p in FONTS_DIR.glob("*.ttf"):
        return p
    raise FileNotFoundError("Hiçbir font bulunamadı (pipeline/fonts klasörü boş).")

def text_to_gokturk_string(text: str) -> str:
    engine, class_meta = _get_engine()
    pairs = engine.expected_sequence_with_letters(text)
    sequence_glyphs = []
    for cid, latin_chunk in pairs:
        if cid == ":":
            sequence_glyphs.append("⁚") # U+205A Word Separator
        elif cid == "literal_colon":
            sequence_glyphs.append(":")
        elif cid in class_meta:
            glyph = (class_meta[cid].get("glyph_ref") or {}).get("core_orhun")
            if glyph:
                sequence_glyphs.append(glyph)
        else:
            sequence_glyphs.append(cid)
    return "".join(sequence_glyphs)

def create_background(size: int | tuple[int, int], style_cfg: dict, background_color: str | None = None) -> Image.Image:
    if isinstance(size, tuple):
        w, h = size
    else:
        w, h = size, size
    max_dim = max(w, h)
        
    bg_color = style_cfg["bg"]
    if background_color:
        try:
            bg_color = ImageColor.getrgb(background_color)
        except Exception:
            pass
            
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)
    rng = random.Random(42) # fixed seed for texture consistency
    
    effects = style_cfg["effects"]
    if "noise" in effects or "grain" in effects or "texture" in effects or "fibers" in effects:
        for _ in range(rng.randint(50, 150)):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(1, 15)
            jitter = rng.randint(-30, 30)
            c = tuple(max(0, min(255, v + jitter)) for v in bg_color)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 3.0)))
        
    if "vignette" in effects or "burn_edges" in effects:
        Y_vg, X_vg = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        cx, cy = w / 2.0, h / 2.0
        max_dist = math.hypot(cx, cy)
        dist = np.hypot(X_vg - cx, Y_vg - cy)
        intensity_arr = np.clip(1.0 - (dist / max_dist), 0.0, 1.0) * 255.0
        vig_img = Image.fromarray(intensity_arr.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=max_dim * 0.1))
        img = Image.composite(img, Image.new("RGB", (w, h), (20, 10, 0)), vig_img)
        
    return img

def render(
    text: str,
    style: str = "plain",
    size: int = 512,
    degradation: float = 0.0,
    font_name: str = "NotoSansOldTurkic-Regular.ttf",
    background_color: str | None = None,
    texture: str | None = None,
    texture_var: str | None = None,
    text_color: str | None = None,
    transparent_bg: bool = False,
    stamp_var: str | None = None,
    light_direction: tuple[int, int] | None = None,
    font_variant: str = "auto",
    watermark: bool = False
) -> Image.Image:
    """Latin/Göktürkçe metni alır, kompozit (doku, gölge, ışık) ile stilize PNG (PIL.Image) döner."""
    # Q2 — Yönsel ışık kaynağı normalizasyonu (Varsayılan Sol-Üst: (-1, -1))
    if not light_direction or not isinstance(light_direction, (tuple, list)) or len(light_direction) < 2:
        lx, ly = -1, -1
    else:
        lx = -1 if light_direction[0] < 0 else (1 if light_direction[0] > 0 else 0)
        ly = -1 if light_direction[1] < 0 else (1 if light_direction[1] > 0 else 0)
        if lx == 0 and ly == 0:
            lx, ly = -1, -1

    # GERİYE DÖNÜK UYUMLULUK (Backward Compatibility):
    legacy_textures = ["stone", "gold", "neon", "wood", "paper", "leather"]
    if style in legacy_textures:
        if texture is None and texture_var is None:
            texture = style + ".png" if not style.endswith(".png") else style
        if style != "neon":
            legacy_fg = STYLES.get(style, {}).get("fg")
            style = "plain"
            style_cfg = dict(STYLES["plain"])
            if legacy_fg and text_color is None:
                style_cfg["fg"] = legacy_fg
        else:
            style_cfg = dict(STYLES["neon"])
    else:
        if style not in STYLES:
            style = "plain"
        style_cfg = dict(STYLES[style])
        
    if stamp_var == "black":
        style_cfg["fg"] = (26, 20, 15)
    if text_color and str(text_color).strip():
        try:
            style_cfg["fg"] = ImageColor.getrgb(str(text_color).strip())
        except Exception:
            pass
            
    lines = [l for l in text.split('\n')]
    if not lines or all(not l.strip() for l in lines):
        lines = [" "]
    line_count = len(lines)
    
    canvas_w = size
    canvas_h = size
        
    # 3-Katmanlı Stil Karar Tablosu ve font_variant Çözümlemesi
    style_spec = resolve_style(style)
    if font_variant and str(font_variant).lower() != "auto":
        v_clean = str(font_variant).strip().capitalize()
        target_font = f"Gokturk-{v_clean}.ttf"
    elif font_name not in ("NotoSansOldTurkic-Regular.ttf", "Gokturk-Regular.ttf", ""):
        target_font = font_name
    else:
        target_font = style_spec["font"]
        
    font_path = get_font_path(target_font, fallback_font=style_spec["fallback"])
    
    margin = int(canvas_w * 0.1)
    max_w = canvas_w - 2 * margin
    
    font_size = size // 2
    pil_font = ImageFont.truetype(str(font_path), font_size)
    
    line_data = []
    max_line_w = 0
    for l_text in lines:
        gokturk_str = text_to_gokturk_string(l_text)
        if not gokturk_str:
            gokturk_str = " "
        try:
            bbox = ImageDraw.Draw(Image.new("RGB", (1,1))).textbbox((0, 0), gokturk_str, font=pil_font, direction='rtl')
            direction = 'rtl'
            text_to_draw = gokturk_str
        except Exception:
            bbox = ImageDraw.Draw(Image.new("RGB", (1,1))).textbbox((0, 0), gokturk_str, font=pil_font)
            direction = None
            text_to_draw = "".join(reversed(gokturk_str))
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        if lw > max_line_w:
            max_line_w = lw
        line_data.append({"text": text_to_draw, "bbox": bbox, "w": lw, "h": lh, "direction": direction, "orig_gok": gokturk_str})
        
    if max_line_w > max_w and max_line_w > 0:
        ratio = max_w / float(max_line_w)
        font_size = int(font_size * ratio)
        pil_font = ImageFont.truetype(str(font_path), font_size)
        for item in line_data:
            gok_str = item["orig_gok"]
            if item["direction"] == 'rtl':
                bbox = ImageDraw.Draw(Image.new("RGB", (1,1))).textbbox((0, 0), gok_str, font=pil_font, direction='rtl')
            else:
                bbox = ImageDraw.Draw(Image.new("RGB", (1,1))).textbbox((0, 0), item["text"], font=pil_font)
            item["bbox"] = bbox
            item["w"] = bbox[2] - bbox[0]
            item["h"] = bbox[3] - bbox[1]
            
    line_spacing = int(font_size * 1.2)
    total_block_h = (line_count - 1) * line_spacing + (line_data[-1]["h"] if line_data else 0)
    
    v_padding = int(max(24.0, font_size * 0.3))
    canvas_h = int(total_block_h + 2 * v_padding)
    if canvas_h % 2 != 0:
        canvas_h += 1
    canvas_h = max(64, canvas_h)
    
    # Base high-precision mask (L1 for plain)
    mask = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(mask)
    
    start_y = (canvas_h - total_block_h) / 2.0 - (line_data[0]["bbox"][1] if line_data else 0)
    
    for i, item in enumerate(line_data):
        x = (canvas_w - item["w"]) / 2.0 - item["bbox"][0]
        y = start_y + i * line_spacing
        if item["direction"] == 'rtl':
            draw.text((x, y), item["text"], fill=255, font=pil_font, direction='rtl')
        else:
            draw.text((x, y), item["text"], fill=255, font=pil_font)
        
    # Texture loading or fallback background
    loaded_texture = False
    tex_file = None
    if transparent_bg:
        bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        loaded_texture = True
    else:
        tex_param = texture or texture_var
        if tex_param:
            candidate = PRODUCT_DIR / "textures" / tex_param
            if not candidate.exists() and not str(tex_param).endswith(".png"):
                candidate = PRODUCT_DIR / "textures" / f"{tex_param}.png"
            if candidate.exists():
                tex_file = candidate
            else:
                variations = get_texture_variants(tex_param)
                if variations:
                    tex_file = random.choice(variations)
        if not tex_file:
            sug = STYLE_TO_TEXTURE_SUGGESTION.get(style)
            if sug:
                sug_name = sug[0] if isinstance(sug, (list, tuple)) else sug
                candidate = PRODUCT_DIR / "textures" / sug_name
                if candidate.exists():
                    tex_file = candidate
                else:
                    variations = get_texture_variants(sug_name)
                    if variations:
                        tex_file = random.choice(variations)
        if not tex_file:
            variations = get_texture_variants(style)
            if variations:
                tex_file = random.choice(variations)
            else:
                texture_path = style_cfg.get("texture_path")
                if texture_path:
                    candidate = PRODUCT_DIR / texture_path
                    if candidate.exists():
                        tex_file = candidate
                    
        if tex_file and tex_file.exists():
            try:
                tex_img = Image.open(tex_file).convert("RGBA")
                tile = tex_img.resize((canvas_w, canvas_w), Image.Resampling.LANCZOS)
                bg = Image.new("RGBA", (canvas_w, canvas_h))
                y_pos = 0
                tile_idx = 0
                while y_pos < canvas_h:
                    t_draw = tile if tile_idx % 2 == 0 else ImageOps.flip(tile)
                    bg.paste(t_draw, (0, y_pos))
                    y_pos += canvas_w
                    tile_idx += 1
                loaded_texture = True
            except Exception:
                pass
                
        if not loaded_texture:
            bg = create_background((canvas_w, canvas_h), style_cfg, background_color).convert("RGBA")
        
    effects = style_cfg["effects"]
    rng_seed = sum(ord(c) for c in text) + size + 42

    # Q3 — ÇOK KATMANLI EFEKT MİMARİSİ (10 STİL)

    # 1. FIRCHA (3 Katman: L1 brush wave+noise, L2 ink load center var, L3 edge bleed)
    if style == "fircha" or "brush" in effects:
        blur_radius = max(1.2, float(font_size) * 0.024)
        blur_img = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blur_arr = np.array(blur_img, dtype=np.float32)
        
        Y_br, X_br = np.meshgrid(np.arange(canvas_h), np.arange(canvas_w), indexing='ij')
        f_scale = 256.0 / max(20.0, float(font_size))
        stroke_wave = np.sin(X_br * 0.02 * f_scale + Y_br * 0.03 * f_scale) * 18.0 + np.cos(X_br * 0.04 * f_scale - Y_br * 0.02 * f_scale) * 10.0
        
        rng_brush = np.random.RandomState(rng_seed)
        low_res_h = max(6, int(canvas_h / max(10.0, float(font_size) * 0.35)))
        low_res_w = max(6, int(canvas_w / max(10.0, float(font_size) * 0.35)))
        noise_grid = rng_brush.normal(0, 1, (low_res_h, low_res_w)).astype(np.float32)
        noise_img = Image.fromarray(noise_grid).resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
        noise_arr = np.array(noise_img) * 14.0
        
        hf_h = max(12, int(canvas_h / max(4.0, float(font_size) * 0.12)))
        hf_w = max(12, int(canvas_w / max(4.0, float(font_size) * 0.12)))
        hf_grid = rng_brush.normal(0, 1, (hf_h, hf_w)).astype(np.float32)
        hf_img = Image.fromarray(hf_grid).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        hf_arr = np.array(hf_img) * 16.0
        
        dynamic_thresh = 128.0 + stroke_wave + noise_arr + hf_arr
        brush_mask_arr = np.clip((blur_arr - dynamic_thresh) * 4.0 + 128.0, 0.0, 255.0)
        
        # L2: Ink load variation (center heavier, edges lighter)
        center_w = np.array(mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, float(font_size) * 0.03))), dtype=np.float32) / 255.0
        brush_mask_arr = brush_mask_arr * (0.65 + 0.35 * center_w)
        
        # L3: Edge bleed
        edge_bleed = np.array(Image.fromarray(brush_mask_arr.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=max(0.5, float(font_size) * 0.008))), dtype=np.float32)
        brush_mask_arr = np.maximum(brush_mask_arr, edge_bleed * 0.4)
        
        mask = Image.fromarray(np.clip(brush_mask_arr, 0, 255).astype(np.uint8), mode="L")

    # 2. INK_BLEED (3 Katman: L1 fiber-aligned, L2 1px capillary spread, L3 edge softening)
    if style == "ink_bleed" or "ink_bleed_effect" in effects:
        blur_radius = max(1.0, float(font_size) * 0.022)
        blur_img = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blur_arr = np.array(blur_img, dtype=np.float32)
        
        rng_bleed = np.random.RandomState(rng_seed + 35)
        fiber_h = max(10, int(canvas_h / max(4.0, float(font_size) * 0.15)))
        fiber_w = max(10, int(canvas_w / max(4.0, float(font_size) * 0.15)))
        fiber_grid = rng_bleed.normal(0, 1, (fiber_h, fiber_w)).astype(np.float32)
        fiber_img = Image.fromarray(fiber_grid).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        fiber_noise = np.array(fiber_img) * 32.0
        
        dynamic_thresh = 128.0 + fiber_noise
        bleed_mask_arr = np.clip((blur_arr - dynamic_thresh) * 5.0 + 128.0, 0.0, 255.0)
        
        # L2: Capillary 1px spread (X and Y directional shift)
        shift_x = np.roll(bleed_mask_arr, 1, axis=1)
        shift_y = np.roll(bleed_mask_arr, 1, axis=0)
        bleed_mask_arr = np.maximum.reduce([bleed_mask_arr, shift_x * 0.7, shift_y * 0.7])
        
        # L3: Edge softening
        soft_bleed = np.array(Image.fromarray(bleed_mask_arr.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=max(0.5, float(font_size) * 0.005))), dtype=np.float32)
        bleed_mask_arr = np.clip(soft_bleed, 0.0, 255.0)
        
        mask = Image.fromarray(bleed_mask_arr.astype(np.uint8), mode="L")

    # 3. STAMP (L1 pressure, L2 grunge - P0 REVERSION, L3 edge wear REMOVED)
    if style == "stamp" or "stamp_effect" in effects:
        rng_stamp_np = np.random.RandomState(rng_seed + 46)
        rng_stamp_py = random.Random(rng_seed + 46)
        
        if stamp_var == "rot":
            rot_angle = rng_stamp_py.uniform(4.0, 5.0) * rng_stamp_py.choice([-1, 1])
        else:
            rot_angle = rng_stamp_py.uniform(-1.5, 1.5)
        mask = mask.rotate(rot_angle, resample=Image.Resampling.BICUBIC, fillcolor=0)
        
        low_res_h = max(4, int(canvas_h / max(10.0, float(font_size) * 0.45)))
        low_res_w = max(4, int(canvas_w / max(10.0, float(font_size) * 0.45)))
        noise_low = rng_stamp_np.normal(0.85, 0.2, (low_res_h, low_res_w)).astype(np.float32)
        noise_low_img = Image.fromarray(noise_low).resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
        
        hf_h = max(10, int(canvas_h / max(3.0, float(font_size) * 0.08)))
        hf_w = max(10, int(canvas_w / max(3.0, float(font_size) * 0.08)))
        noise_hf = rng_stamp_np.normal(0, 1, (hf_h, hf_w)).astype(np.float32)
        noise_hf_img = Image.fromarray(noise_hf).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        hf_arr = np.array(noise_hf_img)
        
        if stamp_var == "grunge":
            pressure_mult = np.clip(np.array(noise_low_img), 0.15, 0.85)
            grunge_holes = np.where(hf_arr > 0.8, 0.05, 1.0)
            grunge_holes = np.where((hf_arr > 0.2) & (hf_arr <= 0.8), 0.35, grunge_holes)
        else:
            pressure_mult = np.clip(np.array(noise_low_img), 0.35, 1.0)
            grunge_holes = np.where(hf_arr > 1.2, 0.15, 1.0)
            grunge_holes = np.where((hf_arr > 0.6) & (hf_arr <= 1.2), 0.65, grunge_holes)
        
        total_grunge = pressure_mult * grunge_holes
        
        mask_arr = np.array(mask, dtype=np.float32)
        stamped_mask_arr = np.clip(mask_arr * total_grunge, 0, 255).astype(np.uint8)
        mask = Image.fromarray(stamped_mask_arr, mode="L")

    # 4. PARCHMENT (3 Katman: L1 fiber absorption, L2 ink bleed gradient, L3 burn edges/vignette)
    if style == "parchment" or "parchment_effect" in effects or "stains" in effects:
        rng_parch_np = np.random.RandomState(rng_seed + 59)
        blur_radius = max(0.8, float(font_size) * 0.015)
        mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        fiber_h = max(10, int(canvas_h / max(4.0, float(font_size) * 0.15)))
        fiber_w = max(10, int(canvas_w / max(4.0, float(font_size) * 0.15)))
        fiber_noise = rng_parch_np.normal(0.92, 0.12, (fiber_h, fiber_w)).astype(np.float32)
        fiber_img = Image.fromarray(fiber_noise).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        
        mask_arr = np.array(mask_blurred, dtype=np.float32)
        parch_mask_arr = np.clip(mask_arr * np.array(fiber_img), 0, 255).astype(np.uint8)
        mask = Image.fromarray(parch_mask_arr, mode="L")

    # 5. CARVED (3 Katman: L1 chisel chipping, L2 directional inner shadow, L3 directional inner highlight)
    if style == "carved" or "carved_effect" in effects:
        rng_carve_np = np.random.RandomState(rng_seed + 66)
        chip_h = max(10, int(canvas_h / max(3.0, float(font_size) * 0.08)))
        chip_w = max(10, int(canvas_w / max(3.0, float(font_size) * 0.08)))
        chip_noise = rng_carve_np.normal(1.0, 0.15, (chip_h, chip_w)).astype(np.float32)
        chip_img = Image.fromarray(chip_noise).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        
        mask_arr = np.array(mask, dtype=np.float32)
        carve_mask_arr = np.clip(mask_arr * np.array(chip_img), 0, 255).astype(np.uint8)
        mask = Image.fromarray(carve_mask_arr, mode="L")

    # 6. CHALK (P2 FIX: Letter-First outer edge erosion only, body untouched)
    if style == "chalk" or "chalk_effect" in effects:
        rng_chalk_np = np.random.RandomState(rng_seed + 101)
        mask_arr = np.array(mask, dtype=np.float32)
        
        chalk_h = max(10, int(canvas_h / max(2.0, float(font_size) * 0.05)))
        chalk_w = max(10, int(canvas_w / max(2.0, float(font_size) * 0.05)))
        noise_chalk = rng_chalk_np.normal(0.7, 0.3, (chalk_h, chalk_w)).astype(np.float32)
        noise_img = Image.fromarray(noise_chalk).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        noise_arr = np.array(noise_img)
        
        # L2: Outer edge erosion only (Body mask_arr >= 150 untouched!)
        eroded_noise = np.where(noise_arr < 0.35, 0.0, noise_arr)
        mask_safe = np.where(mask_arr >= 150.0, mask_arr, mask_arr * eroded_noise)
        
        # L3: Edge dust
        blur_r = max(0.8, float(font_size) * 0.018)
        blurred_mask = np.array(mask.filter(ImageFilter.GaussianBlur(radius=blur_r)), dtype=np.float32)
        edge_powder = np.minimum(blurred_mask * 1.25, mask_safe)
        
        chalk_mask_arr = np.clip(edge_powder, 0, 255).astype(np.uint8)
        mask = Image.fromarray(chalk_mask_arr, mode="L")
        
        # L4: Directional Drop shadow
        offset_val = max(1, int(float(font_size) * 0.02))
        blur_val = max(0.5, float(font_size) * 0.008)
        sh_mask = Image.new("L", (canvas_w, canvas_h), 0)
        sh_mask.paste(mask, (-lx * offset_val, -ly * offset_val))
        sh_mask = sh_mask.filter(ImageFilter.GaussianBlur(radius=blur_val))
        if transparent_bg:
            sh_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            sh_layer.paste((0, 0, 0, 76), (0, 0), sh_mask)
            bg = Image.alpha_composite(bg, sh_layer)
        else:
            sh_layer = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            sh_layer.paste((179, 179, 179), (0, 0), sh_mask)
            bg = ImageChops.multiply(bg.convert("RGB"), sh_layer).convert("RGBA")

    # 7. ASH (P2 FIX: Surface-only cracks, body untouched)
    if style == "ash" or "ash_effect" in effects:
        rng_ash_np = np.random.RandomState(rng_seed + 111)
        blur_ash = max(0.5, float(font_size) * 0.008)
        mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=blur_ash))
        
        # L1: Micro-cracks (3 angles: 0°, 60°, 120°)
        res_h1 = max(10, int(canvas_h / max(2.0, float(font_size) * 0.35)))
        res_w1 = max(10, int(canvas_w / max(2.0, float(font_size) * 0.35)))
        Y_st1, X_st1 = np.meshgrid(np.linspace(0, res_h1, canvas_h), np.linspace(0, res_w1, canvas_w), indexing='ij')
        
        cracks = []
        for angle_deg in [0, 60, 120]:
            theta = np.radians(angle_deg)
            grid_noise = rng_ash_np.normal(0.5, 0.2, (res_h1, res_w1)).astype(np.float32)
            noise_img = Image.fromarray(grid_noise).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
            noise_arr = np.array(noise_img)
            proj = Y_st1 * np.cos(theta) + X_st1 * np.sin(theta)
            wave = 0.5 + 0.3 * np.sin(proj * 1.5 + rng_ash_np.uniform(0, 10))
            combined_noise = (noise_arr * 0.6 + wave * 0.4)
            cracks.append(np.where(np.abs(combined_noise - 0.5) < 0.06, 0.3, 1.0))
            
        # L2: Meso-cracks (2 angles: 90°, 135°)
        res_h2 = max(8, int(canvas_h / max(2.0, float(font_size) * 0.18)))
        res_w2 = max(8, int(canvas_w / max(2.0, float(font_size) * 0.18)))
        Y_st2, X_st2 = np.meshgrid(np.linspace(0, res_h2, canvas_h), np.linspace(0, res_w2, canvas_w), indexing='ij')
        for angle_deg in [90, 135]:
            theta = np.radians(angle_deg)
            grid_noise = rng_ash_np.normal(0.5, 0.2, (res_h2, res_w2)).astype(np.float32)
            noise_img = Image.fromarray(grid_noise).resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
            noise_arr = np.array(noise_img)
            proj = Y_st2 * np.cos(theta) + X_st2 * np.sin(theta)
            wave = 0.5 + 0.3 * np.sin(proj * 1.2 + rng_ash_np.uniform(0, 10))
            combined_noise = (noise_arr * 0.6 + wave * 0.4)
            cracks.append(np.where(np.abs(combined_noise - 0.5) < 0.04, 0.4, 1.0))

        crack_combined = np.minimum.reduce(cracks)
        
        mask_arr = np.array(mask, dtype=np.float32)
        # Body (alpha >= 200) untouched; cracks only applied to surface/edges (alpha < 200)
        surface_crack_mask = np.where(mask_arr >= 200.0, 1.0, crack_combined)
        
        mask_arr = mask_arr * (np.array(mask_blurred, dtype=np.float32) / 255.0 * 0.2 + 0.8) * surface_crack_mask
        ash_mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)
        mask = Image.fromarray(ash_mask_arr, mode="L")
        
        # L4: Dust halo (outer alpha bleed)
        halo_mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, float(font_size) * 0.05)))
        if transparent_bg:
            halo_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            halo_layer.paste((200, 200, 200, 31), (0, 0), halo_mask)
            bg = Image.alpha_composite(bg, halo_layer)
        else:
            halo_layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            halo_layer.paste((31, 31, 31), (0, 0), halo_mask)
            bg = ImageChops.screen(bg.convert("RGB"), halo_layer).convert("RGBA")

    # 8. STENCIL (P2 FIX: Larger period, smaller gap, drop shadow REMOVED)
    if style == "stencil" or "stencil_effect" in effects:
        period_x = max(12.0, float(font_size) * 0.36)
        period_y = max(12.0, float(font_size) * 0.44)
        gap_x = max(1.5, float(font_size) * 0.03)
        gap_y = max(1.5, float(font_size) * 0.035)
        
        Y_st, X_st = np.meshgrid(np.arange(canvas_h), np.arange(canvas_w), indexing='ij')
        cut = ((X_st % period_x) < gap_x) | ((Y_st % period_y) < gap_y)
        stencil_mask = ~cut
        
        mask_arr = np.array(mask, dtype=np.uint8)
        stencil_mask_arr = np.where(stencil_mask, mask_arr, 0).astype(np.uint8)
        mask = Image.fromarray(stencil_mask_arr, mode="L")

    # DERİNLİK EFEKTİ (Diğer stiller için yönsel ışık kaynağı - Q2)
    if style not in ["plain", "neon", "fircha", "ink_bleed", "stamp", "parchment", "carved", "chalk", "ember", "ash", "stencil"] and "brush" not in effects and "ink_bleed_effect" not in effects and "stamp_effect" not in effects and "parchment_effect" not in effects and "carved_effect" not in effects and "chalk_effect" not in effects and "ember_effect" not in effects and "ash_effect" not in effects and "stencil_effect" not in effects:
        offset = max(2, int(size / 256))
        sh_x, sh_y = -lx * offset, -ly * offset
        hi_x, hi_y = lx * offset, ly * offset
        
        shadow_mask = Image.new("L", (canvas_w, canvas_h), 0)
        shadow_mask.paste(mask, (sh_x, sh_y))
        shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, offset * 0.5)))
        
        if transparent_bg:
            shadow_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            shadow_rgba.paste((0, 0, 0, 100), (0, 0), shadow_mask)
            bg = Image.alpha_composite(bg, shadow_rgba)
        else:
            shadow_layer = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            shadow_layer.paste((153, 153, 153), (0, 0), shadow_mask)
            bg = ImageChops.multiply(bg.convert("RGB"), shadow_layer).convert("RGBA")
        
        highlight_mask = Image.new("L", (canvas_w, canvas_h), 0)
        highlight_mask.paste(mask, (hi_x, hi_y))
        highlight_mask = highlight_mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, offset * 0.5)))
        
        if transparent_bg:
            highlight_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            highlight_rgba.paste((255, 255, 255, 76), (0, 0), highlight_mask)
            bg = Image.alpha_composite(bg, highlight_rgba)
        else:
            highlight_layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            highlight_layer.paste((76, 76, 76), (0, 0), highlight_mask)
            bg = ImageChops.screen(bg.convert("RGB"), highlight_layer).convert("RGBA")

        if style == "gold":
            gold_mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, offset * 0.8)))
            if transparent_bg:
                gold_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                gold_rgba.paste((255, 215, 0, 150), (0, 0), gold_mask)
                bg = Image.alpha_composite(bg, gold_rgba)
            else:
                gold_layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
                gold_layer.paste((100, 80, 0), (0, 0), gold_mask)
                bg = ImageChops.screen(bg.convert("RGB"), gold_layer).convert("RGBA")

    # 9. NEON 
    # LOCKED - kullanıcı onayı, değiştirme
    fg_color = style_cfg["fg"]
    if style == "neon" or "glow" in effects or "bloom" in effects:
        glow_mask1 = mask.filter(ImageFilter.GaussianBlur(radius=size*0.04))
        glow_layer1 = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
        glow_layer1.paste(fg_color, (0, 0), glow_mask1)
        bg = Image.alpha_composite(bg, glow_layer1)
        
        glow_mask2 = mask.filter(ImageFilter.GaussianBlur(radius=size*0.02))
        glow_layer2 = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
        glow_layer2.paste(fg_color, (0, 0), glow_mask2)
        bg = Image.alpha_composite(bg, glow_layer2)
        
        glow_mask3 = mask.filter(ImageFilter.GaussianBlur(radius=size*0.01))
        glow_layer3 = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
        glow_layer3.paste((255, 255, 255, 255), (0, 0), glow_mask3)
        bg = Image.alpha_composite(bg, glow_layer3)
        
        core_layer = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
        core_layer.paste((255, 255, 255, 255), (0, 0), mask)
        bg = Image.alpha_composite(bg, core_layer)

    # 10. EMBER (P2 FIX: L4 hot white core REMOVED, vibrant orange preserved)
    blend_mode = style_cfg.get("blend_mode", "normal")
    if style == "ember" or "ember_effect" in effects:
        # L1 — Outer halo
        glow_r1 = max(5.0, float(font_size) * 0.10)
        glow_mask1 = mask.filter(ImageFilter.GaussianBlur(radius=glow_r1))
        glow_layer1 = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_layer1.paste((180, 30, 0, 100), (0, 0), glow_mask1)
        bg = Image.alpha_composite(bg, glow_layer1)
        
        # L2 — Middle glow
        glow_r2 = max(2.5, float(font_size) * 0.05)
        glow_mask2 = mask.filter(ImageFilter.GaussianBlur(radius=glow_r2))
        glow_layer2 = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        glow_layer2.paste((255, 120, 20, 150), (0, 0), glow_mask2)
        bg = Image.alpha_composite(bg, glow_layer2)
        
        # L3 — Inner core gradient (vibrant orange, upper bound capped so center stays orange!)
        blur_ember = max(1.0, float(font_size) * 0.025)
        center_weight = np.array(mask.filter(ImageFilter.GaussianBlur(radius=blur_ember)), dtype=np.float32) / 255.0
        
        fg_r, fg_g, fg_b = fg_color
        emb_r = np.clip(fg_r * 0.25 + center_weight * (240.0 - 60.0), 0, 255).astype(np.uint8)
        emb_g = np.clip(fg_g * 0.10 + center_weight * (210.0 - 60.0), 0, 255).astype(np.uint8)
        emb_b = np.clip(fg_b * 0.05 + center_weight * (90.0 - 60.0), 0, 255).astype(np.uint8)
        emb_a = np.array(mask, dtype=np.uint8)
        ember_rgba = Image.fromarray(np.dstack([emb_r, emb_g, emb_b, emb_a]), mode="RGBA")
        bg = Image.alpha_composite(bg, ember_rgba)
        
    elif style in ["fircha", "ink_bleed", "stamp", "parchment", "carved"] or any(e in effects for e in ["brush", "ink_bleed", "ink_bleed_effect", "stamp_effect", "parchment_effect", "carved_effect"]):
        rng_ink = np.random.RandomState(rng_seed + 100)
        ink_res_h = max(8, int(canvas_h / max(10.0, float(font_size) * 0.25)))
        ink_res_w = max(8, int(canvas_w / max(10.0, float(font_size) * 0.25)))
        ink_grid = rng_ink.normal(0, 1, (ink_res_h, ink_res_w)).astype(np.float32)
        ink_noise = Image.fromarray(ink_grid).resize((canvas_w, canvas_h), Image.Resampling.BICUBIC)
        ink_noise_arr = np.array(ink_noise) * 20.0
        
        Y_ink, X_ink = np.meshgrid(np.arange(canvas_h), np.arange(canvas_w), indexing='ij')
        ink_grad = ((X_ink - Y_ink) / max(canvas_w, canvas_h)) * 35.0
        
        fg_r, fg_g, fg_b = fg_color
        ink_r = np.clip(fg_r + ink_grad + ink_noise_arr, 0, 255).astype(np.uint8)
        ink_g = np.clip(fg_g + ink_grad + ink_noise_arr, 0, 255).astype(np.uint8)
        ink_b = np.clip(fg_b + ink_grad + ink_noise_arr, 0, 255).astype(np.uint8)
        ink_a = np.array(mask, dtype=np.uint8)
        
        ink_rgba = Image.fromarray(np.dstack([ink_r, ink_g, ink_b, ink_a]), mode="RGBA")
        
        if transparent_bg:
            bg = Image.alpha_composite(bg, ink_rgba)
        elif blend_mode == "multiply":
            white_bg = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
            white_bg = Image.alpha_composite(white_bg, ink_rgba)
            bg = ImageChops.multiply(bg.convert("RGB"), white_bg.convert("RGB")).convert("RGBA")
        else:
            bg = Image.alpha_composite(bg, ink_rgba)
    elif style != "neon":
        if transparent_bg:
            text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            text_layer.paste(fg_color, (0, 0), mask)
            bg = Image.alpha_composite(bg, text_layer)
        elif blend_mode == "multiply":
            text_layer = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            text_layer.paste(fg_color, (0, 0), mask)
            bg = ImageChops.multiply(bg.convert("RGB"), text_layer).convert("RGBA")
        elif blend_mode == "screen":
            text_layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            text_layer.paste(fg_color, (0, 0), mask)
            bg = ImageChops.screen(bg.convert("RGB"), text_layer).convert("RGBA")
        else:
            text_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            text_layer.paste(fg_color, (0, 0), mask)
            bg = Image.alpha_composite(bg, text_layer)
        
    # CARVED INNER BEVEL (Q2 directional lighting)
    if style == "carved" or "carved_effect" in effects:
        offset = max(2, int(float(font_size) * 0.04))
        blur_r = max(0.8, float(font_size) * 0.012)
        sh_x, sh_y = -lx * offset, -ly * offset
        hi_x, hi_y = lx * offset, ly * offset
        
        shifted_sh = Image.new("L", (canvas_w, canvas_h), 0)
        shifted_sh.paste(mask, (sh_x, sh_y))
        inner_sh = ImageChops.subtract(mask, shifted_sh).filter(ImageFilter.GaussianBlur(radius=blur_r))
        
        shifted_hi = Image.new("L", (canvas_w, canvas_h), 0)
        shifted_hi.paste(mask, (hi_x, hi_y))
        inner_hi = ImageChops.subtract(mask, shifted_hi).filter(ImageFilter.GaussianBlur(radius=blur_r))
        
        if transparent_bg:
            sh_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            sh_layer.paste((0, 0, 0, 180), (0, 0), inner_sh)
            bg = Image.alpha_composite(bg, sh_layer)
            
            hi_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            hi_layer.paste((255, 255, 255, 140), (0, 0), inner_hi)
            bg = Image.alpha_composite(bg, hi_layer)
        else:
            sh_layer = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            sh_layer.paste((30, 30, 30), (0, 0), inner_sh)
            bg = ImageChops.multiply(bg.convert("RGB"), sh_layer).convert("RGBA")
            
            hi_layer = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            hi_layer.paste((200, 200, 200), (0, 0), inner_hi)
            bg = ImageChops.screen(bg.convert("RGB"), hi_layer).convert("RGBA")
        
    # Degradation > 0.2
    if degradation > 0.2 and not transparent_bg:
        deg = min(1.0, degradation)
        rng = random.Random()
        bg_rgb = bg.convert("RGB")
        if rng.random() < deg:
            bg_rgb = bg_rgb.filter(ImageFilter.GaussianBlur(radius=deg * 2.0))
        if rng.random() < deg:
            enhancer = ImageEnhance.Brightness(bg_rgb)
            bg_rgb = enhancer.enhance(1.0 + rng.uniform(-0.3, 0.3) * deg)
        if rng.random() < deg:
            enhancer = ImageEnhance.Contrast(bg_rgb)
            bg_rgb = enhancer.enhance(1.0 + rng.uniform(-0.4, 0.4) * deg)
        bg = bg_rgb.convert("RGBA")
        
    return bg if transparent_bg else bg.convert("RGB")

