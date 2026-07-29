import os
import sys
import math
import copy
from pathlib import Path
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.subset import main as subset_main

FONTS_DIR = Path(r"C:\Users\pc\gokturk_studio\pipeline\fonts")
BASE_FONT_PATH = FONTS_DIR / "NotoSansOldTurkic-Regular.ttf"

def change_weight(font: TTFont, delta: float) -> TTFont:
    """True outline weight modification using contour normal vector offsets."""
    font = copy.deepcopy(font)
    glyf = font['glyf']
    hmtx = font['hmtx']
    
    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.numberOfContours > 0 and hasattr(g, 'coordinates'):
            coords = np.array(g.coordinates, dtype=np.float64)
            end_pts = g.endPtsOfContours
            new_coords = coords.copy()
            
            start_idx = 0
            for end_idx in end_pts:
                c_len = end_idx - start_idx + 1
                if c_len >= 3:
                    c = coords[start_idx:end_idx + 1]
                    prev_pts = np.roll(c, 1, axis=0)
                    next_pts = np.roll(c, -1, axis=0)
                    
                    tangents = next_pts - prev_pts
                    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
                    norms = np.linalg.norm(normals, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    unit_normals = normals / norms
                    
                    offsets = unit_normals * delta
                    new_coords[start_idx:end_idx + 1] += offsets
                start_idx = end_idx + 1
                
            for i in range(len(g.coordinates)):
                g.coordinates[i] = (int(round(new_coords[i, 0])), int(round(new_coords[i, 1])))
            g.recalcBounds(glyf)
            
            w, lsb = hmtx[gname]
            hmtx[gname] = (max(0, int(round(w + delta * 1.2))), int(round(lsb)))
            
    return font

def apply_oblique(font: TTFont, angle_deg: float = 14.0) -> TTFont:
    """Shear transformation for Oblique/Italic slant."""
    font = copy.deepcopy(font)
    glyf = font['glyf']
    slant = math.tan(math.radians(angle_deg))
    matrix = ((1, 0), (slant, 1))
    
    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.numberOfContours > 0 and hasattr(g, 'coordinates'):
            g.coordinates.transform(matrix)
            g.recalcBounds(glyf)
            
    return font

def apply_condense(font: TTFont, scale_x: float = 0.75) -> TTFont:
    """Horizontal scaling transform for Condensed width."""
    font = copy.deepcopy(font)
    glyf = font['glyf']
    hmtx = font['hmtx']
    matrix = ((scale_x, 0), (0, 1.0))
    
    for gname in font.getGlyphOrder():
        g = glyf[gname]
        if g.numberOfContours > 0 and hasattr(g, 'coordinates'):
            g.coordinates.transform(matrix)
            g.recalcBounds(glyf)
            
        w, lsb = hmtx[gname]
        hmtx[gname] = (int(round(w * scale_x)), int(round(lsb * scale_x)))
        
    return font

def build_all_fonts():
    print(f"Loading base font: {BASE_FONT_PATH}")
    base_font = TTFont(str(BASE_FONT_PATH))
    
    variants = {
        "NotoSansOldTurkic-Regular.ttf": base_font,
        "NotoSansOldTurkic-Bold.ttf": change_weight(base_font, delta=35.0),
        "NotoSansOldTurkic-Light.ttf": change_weight(base_font, delta=-20.0),
        "NotoSansOldTurkic-Oblique.ttf": apply_oblique(base_font, angle_deg=14.0),
        "NotoSansOldTurkic-Condensed.ttf": apply_condense(base_font, scale_x=0.75),
        "NotoSansOldTurkic-BoldCondensed.ttf": apply_condense(change_weight(base_font, delta=35.0), scale_x=0.75),
        "NotoSansOldTurkic-BoldOblique.ttf": apply_oblique(change_weight(base_font, delta=35.0), angle_deg=14.0),
    }
    
    print("\n1. Saving derived full TTF variants...")
    temp_files = []
    for filename, f_obj in variants.items():
        out_path = FONTS_DIR / filename
        f_obj.save(str(out_path))
        temp_files.append(out_path)
        print(f"   -> Saved {filename} ({out_path.stat().st_size / 1024:.1f} KB)")
        
    print("\n2. Subsetting to U+10C00-10C4F + required glyphs (.ttf and .woff2)...")
    unicodes_str = "U+10C00-10C4F,U+0020,U+205A,U+003A"
    
    for filename in variants.keys():
        ttf_path = FONTS_DIR / filename
        woff2_name = filename.replace(".ttf", ".woff2")
        woff2_path = FONTS_DIR / woff2_name
        
        # Subsetting TTF in-place
        subset_args_ttf = [
            str(ttf_path),
            f"--unicodes={unicodes_str}",
            f"--output-file={ttf_path}"
        ]
        subset_main(subset_args_ttf)
        
        # Subsetting WOFF2
        subset_args_woff2 = [
            str(ttf_path),
            f"--unicodes={unicodes_str}",
            "--flavor=woff2",
            f"--output-file={woff2_path}"
        ]
        subset_main(subset_args_woff2)
        
        ttf_kb = ttf_path.stat().st_size / 1024
        woff2_kb = woff2_path.stat().st_size / 1024
        print(f"   -> Subsetted {filename:35s} TTF: {ttf_kb:5.1f} KB | WOFF2: {woff2_kb:5.1f} KB")

    print("\nAll 7 font variants (.ttf and .woff2) generated and subsetted successfully!")

if __name__ == "__main__":
    build_all_fonts()
