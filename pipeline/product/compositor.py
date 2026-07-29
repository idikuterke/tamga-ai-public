"""
pipeline/product/compositor.py
================================
Göktürkçe Doğrulama Aracı - Compositör Modülü
================================
AI görsellerin üzerine, %100 doğru Göktürkçe metni, hedef bölgenin
renk/ışık/perspektifine uyarlayarak "yapıştırılmış" hissini en aza
indirerek yerleştirir.

Kullanım:
    python compositor.py --base ai_background.png --text "𐱅𐰭𐰼𐰃" --bbox 100,200,400,100 --style stone
"""
from __future__ import annotations
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
from typing import Tuple, Optional, Dict
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BBox = Tuple[int, int, int, int]
Corners = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
RGBA = Image.Image


def _bbox_to_corners(bbox: BBox) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def analyze_region(image: RGBA, bbox: BBox) -> Dict:
    x, y, w, h = bbox
    region = image.crop(_bbox_to_corners(bbox)).convert("RGB")
    arr = np.array(region)
    avg_color = tuple(int(c) for c in arr.mean(axis=(0, 1)))
    gray = arr.mean(axis=2)
    light_y, light_x = np.unravel_index(gray.argmax(), gray.shape)
    dark_y, dark_x = np.unravel_index(gray.argmin(), gray.shape)
    light_dir = (light_x - dark_x, light_y - dark_y)
    variance = float(gray.std())
    is_dark = float(gray.mean()) < 100
    return {"avg_color": avg_color, "light_dir": light_dir, "variance": variance, "is_dark": is_dark}


def render_gokturkce_text(text: str, style: str, size: Tuple[int, int], target_lighting: Optional[Dict] = None, text_color: Optional[str] = None, scale: float = 0.8, texture: Optional[str] = None) -> RGBA:
    """
    ÖNEMLİ: import satırı DÜZELTİLDİ — hem script hem modül olarak çalıştırıldığında
    gerçek render.py'yi bulabilsin. Kullanıcının ölçek (scale) ve renk (text_color)
    seçimi desteklenir.
    """
    try:
        try:
            from render import render  # script olarak çalıştırılırsa
        except ImportError:
            from .render import render  # paket olarak import edilirse
        img = render(text=text, style=style, size=size[0], transparent_bg=True, text_color=text_color, texture=texture)
        
        # 1. Metnin gerçek en-boy oranını bul (boş şeffaf alanları kırp)
        alpha_bbox = img.split()[3].getbbox()
        if alpha_bbox:
            img = img.crop(alpha_bbox)
            
        # 2. Zorla sıkıştırma (squishing/stretching) YASAK. 
        # Bunun yerine bbox içine SIĞDIR (contain/fit, orantılı ölçekle).
        target_w, target_h = size
        scale_ratio = min(target_w / float(img.width), target_h / float(img.height)) * max(0.1, min(2.0, float(scale)))
        new_w = max(1, int(img.width * scale_ratio))
        new_h = max(1, int(img.height * scale_ratio))
        
        if (new_w, new_h) != img.size:
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        # 3. Hedef boyuttaki kanvasın ortasına yerleştir
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(img, (paste_x, paste_y), img)
        return canvas
    except ImportError:
        print("[Uyarı] render.py bulunamadı, fallback PIL renderer kullanılıyor.")
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_paths = [
            "fonts/NotoSansOldTurkic-Regular.ttf",
            "C:/Windows/Fonts/NotoSansOldTurkic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansOldTurkic-Regular.ttf",
        ]
        font = None
        font_size = max(10, int(min(size) * 0.5 * max(0.1, min(2.0, float(scale)))))
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
        if text_color:
            from PIL import ImageColor
            try:
                color = ImageColor.getrgb(str(text_color).strip())
            except Exception:
                color = (200, 200, 200)
        else:
            color = target_lighting["avg_color"] if target_lighting else (200, 200, 200)
            if target_lighting and target_lighting["is_dark"]:
                color = tuple(min(255, c + 100) for c in color)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        draw.text(((size[0] - text_w) // 2, (size[1] - text_h) // 2), text, font=font, fill=color)
        return img


def warp_to_perspective(image: RGBA, src_corners: Corners, dst_corners: Corners) -> RGBA:
    matrix = []
    for (sx, sy), (dx, dy) in zip(src_corners, dst_corners):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array([c[0] for c in src_corners] + [c[1] for c in src_corners], dtype=np.float64).reshape(8)
    try:
        res = np.linalg.solve(A, B)
        coeffs = np.array(res).reshape(8).tolist()
        return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    except np.linalg.LinAlgError:
        return image


def match_color_tone(text_rgba: RGBA, target_region: RGBA) -> RGBA:
    """
    İki adımlı HSV Renk Uyumu:
    a) Metnin renginden bağımsız olarak nötr/gri bir temel forma indirgendiğinde
       sadece şekil, gölge ve parlaklık/relief detayı (Value kanalı) kalır.
    b) Hedef zemin bölgesinin ortalama Hue (H - ton) ve Saturation (S - doygunluk)
       değerleri alınır; metnin kendi Value (V) kanalı korunarak sadece H/S hedefe uyar.
    """
    try:
        text_rgb = text_rgba.convert("RGB")
        text_gray = text_rgb.convert("L").convert("RGB")
        text_hsv_arr = np.array(text_gray.convert("HSV"))
        
        target_rgb = tuple(int(round(c)) for c in np.array(target_region.convert("RGB")).mean(axis=(0, 1)))
        target_hsv_pixel = Image.new("RGB", (1, 1), target_rgb).convert("HSV").getpixel((0, 0))
        target_h, target_s, _ = target_hsv_pixel
        
        text_hsv_arr[:, :, 0] = target_h
        text_hsv_arr[:, :, 1] = target_s
        
        matched_rgb = Image.fromarray(text_hsv_arr, "HSV").convert("RGB")
        result = Image.new("RGBA", text_rgba.size)
        result.paste(matched_rgb, (0, 0))
        if len(text_rgba.split()) == 4:
            result.putalpha(text_rgba.split()[3])
        return result
    except Exception as e:
        print(f"[Uyarı] HSV renk uyumu başarısız: {e}, orijinal dönülüyor.")
        return text_rgba


def synthesize_shadow_and_highlight(text_rgba: RGBA, light_dir: Tuple[int, int], variance: float) -> RGBA:
    w, h = text_rgba.size
    dx, dy = light_dir
    length = max((dx**2 + dy**2) ** 0.5, 1)
    offset_x = int(15 * dx / length)
    offset_y = int(15 * dy / length)
    shadow_alpha = np.array(text_rgba)[:, :, 3]
    shadow_arr = np.zeros((h, w, 4), dtype=np.uint8)
    shadow_arr[:, :, 3] = (shadow_alpha * 0.4).astype(np.uint8)
    shadow_img = Image.fromarray(shadow_arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=4))
    highlight_arr = np.zeros((h, w, 4), dtype=np.uint8)
    highlight_arr[:, :, 3] = (shadow_alpha * 0.3).astype(np.uint8)
    highlight_arr[:, :, :3] = 255
    highlight_img = Image.fromarray(highlight_arr, "RGBA")
    pad_x, pad_y = abs(offset_x) + 10, abs(offset_y) + 10
    canvas_w, canvas_h = w + pad_x * 2, h + pad_y * 2
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(shadow_img, (pad_x + offset_x, pad_y + offset_y), shadow_img)
    canvas.paste(text_rgba, (pad_x, pad_y), text_rgba)
    canvas.paste(highlight_img, (pad_x - offset_x // 2, pad_y - offset_y // 2), highlight_img)
    return canvas.crop((pad_x, pad_y, pad_x + w, pad_y + h))


def composite_text(base_image: RGBA, text: str, bbox: BBox, style: str = "stone",
                    perspective_corners: Optional[Corners] = None, shadow: bool = True,
                    color_match: bool = True, text_color: Optional[str] = None,
                    scale: float = 0.8, auto_color_match: bool = True,
                    texture: Optional[str] = None) -> RGBA:
    if auto_color_match and color_match:
        target_info = analyze_region(base_image, bbox)
        color_to_render = None
    else:
        target_info = None
        color_to_render = text_color

    text_size = (bbox[2], bbox[3])
    text_rgba = render_gokturkce_text(text, style=style, size=text_size, target_lighting=target_info, text_color=color_to_render, scale=scale, texture=texture)
    if perspective_corners:
        src = ((0, 0), (text_size[0], 0), (text_size[0], text_size[1]), (0, text_size[1]))
        text_rgba = warp_to_perspective(text_rgba, src, perspective_corners)
    if auto_color_match and color_match:
        target_region = base_image.crop(_bbox_to_corners(bbox))
        text_rgba = match_color_tone(text_rgba, target_region)
    if shadow:
        light_dir = target_info["light_dir"] if target_info else (-1, -1)
        variance = target_info["variance"] if target_info else 30.0
        text_rgba = synthesize_shadow_and_highlight(text_rgba, light_dir=light_dir, variance=variance)
    result = base_image.copy().convert("RGBA")
    paste_x = bbox[0] + (bbox[2] - text_rgba.width) // 2
    paste_y = bbox[1] + (bbox[3] - text_rgba.height) // 2
    if perspective_corners:
        paste_x, paste_y = bbox[0], bbox[1]
    result.paste(text_rgba, (paste_x, paste_y), text_rgba)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Göktürkçe Compositör Test")
    parser.add_argument("--base", help="Temel görsel dosyası (opsiyonel)")
    parser.add_argument("--text", default="𐱅𐰭𐰼𐰃", help="Göktürkçe metin (Unicode)")
    parser.add_argument("--bbox", default="100,100,400,150", help="x,y,w,h")
    parser.add_argument("--style", default="stone", help="stone, parchment, neon, wood, ink")
    parser.add_argument("--text-color", default=None, help="Özel yazı rengi (#RRGGBB)")
    parser.add_argument("--scale", type=float, default=0.8, help="Metin ölçeği (ör. 0.8)")
    parser.add_argument("--no-auto-color", action="store_true", help="Otomatik renk uyumunu kapat")
    parser.add_argument("--output", default="test_composit.png", help="Çıktı dosyası")
    parser.add_argument("--flat", action="store_true", help="Perspektif ve gölge olmadan düz test")
    args = parser.parse_args()
    bbox = tuple(int(v) for v in args.bbox.split(","))
    if args.base and os.path.exists(args.base):
        base = Image.open(args.base).convert("RGBA")
    else:
        print("Temel görsel belirtilmedi, gerçek taş dokulu temiz zemin yükleniyor...")
        tex_path = os.path.join(os.path.dirname(__file__), "textures", "stone.png")
        if os.path.exists(tex_path):
            base = Image.open(tex_path).convert("RGBA").resize((800, 600), Image.Resampling.LANCZOS)
        else:
            base = Image.new("RGBA", (800, 600), (180, 175, 165))
            
    print(f"Compositing: '{args.text}' -> {args.style} stili, bbox={bbox}, flat={args.flat}, auto_color={not args.no_auto_color}")
    
    corners = None if args.flat else ((bbox[0]+20, bbox[1]), (bbox[0]+bbox[2]-20, bbox[1]+10),
                                      (bbox[0]+bbox[2]-40, bbox[1]+bbox[3]), (bbox[0], bbox[1]+bbox[3]-10))
    result = composite_text(
        base_image=base, text=args.text, bbox=bbox, style=args.style,
        perspective_corners=corners,
        shadow=not args.flat, color_match=not args.flat and not args.no_auto_color,
        text_color=args.text_color, scale=args.scale, auto_color_match=not args.no_auto_color
    )
    result.save(args.output)
    print(f"✅ Başarılı! Sonuç '{args.output}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
