import hashlib
import os
import sys
import tempfile
from typing import Optional, Tuple

import imageio
import numpy as np
from PIL import Image
from transformers import pipeline


class VideoRenderer:
    """
    Göktürkçe görsel çıktılarını statik PNG'den dinamik 3D Parallax veya 
    kamera hareketli MP4 videolara dönüştürür. Depth Anything V2 Small 
    modelini kullanarak monoküler derinlik haritası çıkarır.
    
    Sinematik kalite iyileştirmeleri (Jitter-Free Architecture):
    1. Güvenli Kenar Boşluğu (%15 Safe Padding): Kenar çarpma ve taşmalarını engeller.
    2. LANCZOS Resampling: Yüksek kaliteli alt-piksel (sub-pixel) enterpolasyon.
    3. Matematiksel Yumuşatma (Smoothstep Easing): Ani hız sıçramalarını (jitter) engeller.
    4. Statik Degradasyon Koruması: Frame başına gürültü (karıncalanma) oluşturulmaz.
    """
    def __init__(self):
        self.depth_model = None
        self.cache = {}  # Görsel hash -> depth map cache

    def get_depth_map(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Görselin hash değerini hesaplar; cache'de varsa döndürür, yoksa 
        Depth Anything V2 modeli ile derinlik haritası çıkarır.
        Model yüklenemezse fallback için None döndürür.
        """
        try:
            img_bytes = image.tobytes()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in self.cache:
                return self.cache[img_hash]

            if self.depth_model is None:
                try:
                    sys.stderr.write("Depth Anything V2 modeli yükleniyor...\n")
                    self.depth_model = pipeline(
                        "depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf"
                    )
                except Exception as e:
                    sys.stderr.write(f"Depth model indirilemedi/yüklenemedi (fallback aktif): {e}\n")
                    return None

            res = self.depth_model(image)
            depth_img = res["depth"] if isinstance(res, dict) and "depth" in res else res
            
            # Basit cache yönetimi (maksimum 50 girdi)
            if len(self.cache) >= 50:
                self.cache.clear()
            self.cache[img_hash] = depth_img
            return depth_img
        except Exception as e:
            sys.stderr.write(f"Derinlik haritası çıkarılırken hata oluştu (fallback aktif): {e}\n")
            return None

    def apply_3d_parallax(
        self,
        work_img: Image.Image,
        work_depth: Image.Image,
        frame_idx: int,
        total_frames: int,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Derinlik haritasına göre katmanları farklı hızlarda kaydırır (RTL Okuma Yönü).
        - %15 güvenli kenar boşluğu (safe padding) ile büyütülmüş work_img üzerinden çalışır.
        - LANCZOS yüksek kaliteli enterpolasyon ve sub-pixel doğrusal ağlama (linear interpolation)
          kullanarak tam sayı sıçramalarından kaynaklı titremeleri (jitter) yok eder.
        - Smoothstep (3t^2 - 2t^3) yumuşatma fonksiyonu ile ani hareket sıçramalarını önler.
        - RTL (Sağdan Sola) okuma yönü için kamera sağdan başlar ve zamanla sola doğru kayar.
          Yakın nesneler (metin) uzaktaki dokuya kıyasla daha hızlı sola kayar.
        """
        pad_w, pad_h = work_img.size
        if work_depth.size != (pad_w, pad_h):
            work_depth = work_depth.resize((pad_w, pad_h), Image.Resampling.LANCZOS)

        img_arr = np.array(work_img).astype(float)
        depth_array = np.array(work_depth).astype(float) / 255.0

        # Maksimum piksel kayması
        max_offset = max(12.0, float(target_size[0]) * 0.06)
        
        # 3. Matematiksel Yumuşatma (Smoothstep Easing Function)
        t = frame_idx / max(1, total_frames - 1)
        smooth_t = t * t * (3.0 - 2.0 * t)

        # RTL (Sağdan Sola) Pan ve 3D Parallax:
        # Başlangıçta (smooth_t=0) offset maksimumdur (sağ kenarda), zamanla 0'a (sola) doğru kayar.
        # Yakın nesneler (depth_array ~ 1.0) daha hızlı sola kayar.
        offset_x_float = (1.0 - smooth_t) * max_offset * (0.25 + 0.75 * depth_array)

        y_indices = np.arange(pad_h)[:, None]
        x_indices = np.arange(pad_w)[None, :]
        x_float = np.clip(x_indices + offset_x_float, 0.0, float(pad_w - 1))

        # Sub-pixel doğrusal enterpolasyon (tam sayı sıçramalarını / titremeyi yok eder)
        x0 = np.floor(x_float).astype(int)
        x1 = np.minimum(x0 + 1, pad_w - 1)
        weight = (x_float - x0)[:, :, None]

        warped_arr = np.clip(img_arr[y_indices, x0] * (1.0 - weight) + img_arr[y_indices, x1] * weight, 0, 255).astype(np.uint8)
        warped_img = Image.fromarray(warped_arr)

        # Hedef boyuta crop et (kenarlardaki %15 güvenli padding taşmalarını kes)
        target_w, target_h = target_size
        x0_crop = (pad_w - target_w) // 2
        y0_crop = (pad_h - target_h) // 2
        return warped_img.crop((x0_crop, y0_crop, x0_crop + target_w, y0_crop + target_h))

    def apply_zoom(
        self,
        work_img: Image.Image,
        frame_idx: int,
        total_frames: int,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        %15 güvenli kenar boşluğu (safe padding) ile büyütülmüş work_img üzerinden
        sub-pixel BICUBIC resampling ve smoothstep easing ile titreşimsiz yakınlaştırma.
        """
        W, H = target_size
        big_w, big_h = work_img.size
        
        t = frame_idx / max(1, total_frames - 1)
        smooth_t = t * t * (3.0 - 2.0 * t)
        
        # %15 padding içinden zoom penceresi
        zoom_factor = 1.0 + smooth_t * 0.12
        curr_w = float(W) / zoom_factor
        curr_h = float(H) / zoom_factor
        
        cx, cy = big_w / 2.0, big_h / 2.0
        left = cx - curr_w / 2.0
        top = cy - curr_h / 2.0
        right = cx + curr_w / 2.0
        bottom = cy + curr_h / 2.0
        
        # Image.Transform.EXTENT float (sub-pixel) koordinatlar ve BICUBIC ile sıfır titreme sağlar
        return work_img.transform((W, H), Image.Transform.EXTENT, (left, top, right, bottom), resample=Image.Resampling.BICUBIC)

    def apply_pan(
        self,
        work_img: Image.Image,
        frame_idx: int,
        total_frames: int,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        %15 güvenli kenar boşluğu (safe padding) ile büyütülmüş work_img üzerinden
        sub-pixel BICUBIC resampling ve smoothstep easing ile titreşimsiz RTL kaydırma (pan).
        """
        W, H = target_size
        big_w, big_h = work_img.size
        
        t = frame_idx / max(1, total_frames - 1)
        smooth_t = t * t * (3.0 - 2.0 * t)
        
        max_x = float(big_w - W)
        # RTL Okuma Yönü: Kamera sağdan başlar (max_x) ve zamanla sola (0'a) doğru kayar
        left = (1.0 - smooth_t) * max_x
        top = float((big_h - H) / 2.0)
        right = left + float(W)
        bottom = top + float(H)
        
        return work_img.transform((W, H), Image.Transform.EXTENT, (left, top, right, bottom), resample=Image.Resampling.BICUBIC)

    def apply_fade(self, image: Image.Image, frame_idx: int, total_frames: int) -> Image.Image:
        t = frame_idx / max(1, total_frames - 1)
        smooth_t = t * t * (3.0 - 2.0 * t)
        if smooth_t < 0.15:
            alpha = smooth_t / 0.15
        elif smooth_t > 0.85:
            alpha = (1.0 - smooth_t) / 0.15
        else:
            alpha = 1.0
        img_arr = np.array(image).astype(float)
        faded_arr = np.clip(img_arr * alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(faded_arr)

    def encode_frames(self, frames: list, fps: int = 30) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_name = f.name
        try:
            np_frames = [np.array(fr.convert("RGB")) for fr in frames]
            try:
                imageio.mimsave(temp_name, np_frames, fps=fps, codec="libx264")
            except Exception:
                imageio.mimsave(temp_name, np_frames, fps=fps)
            with open(temp_name, "rb") as f:
                video_bytes = f.read()
            return video_bytes
        finally:
            if os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except Exception:
                    pass

    def render_to_video(self, image: Image.Image, motion: str = "parallax", duration: int = 5, fps: int = 30) -> bytes:
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            bg_solid = Image.new("RGB", image.size, (255, 255, 255))
            bg_solid.paste(image, (0, 0), image.convert("RGBA"))
            image = bg_solid
        else:
            image = image.convert("RGB")

        total_frames = max(1, duration * fps)
        frames = []

        depth_map = None
        if motion == "parallax":
            depth_map = self.get_depth_map(image)
            if depth_map is None:
                sys.stderr.write("Depth model bulunamadı/yüklenemedi, 2D pan hareketine dönülüyor.\n")
                motion = "pan"

        W, H = image.size

        # 1. Güvenli Kenar Boşluğu (Safe Padding %15):
        # Temel görseli %15 büyüterek işleme alıyoruz (en az 1.15x)
        # Böylece pan, zoom veya parallax sırasında kenarlarda taşma veya titreme (jitter) olmaz.
        # 2. Yüksek Kaliteli Resampling: Tüm boyut değiştirmeler açıkça LANCZOS ile yapılır.
        pad_w, pad_h = int(W * 1.15), int(H * 1.15)
        work_img = image.resize((pad_w, pad_h), Image.Resampling.LANCZOS)
        
        work_depth = None
        if motion == "parallax" and depth_map is not None:
            work_depth = depth_map.resize((pad_w, pad_h), Image.Resampling.LANCZOS)

        # 4. Degradasyon Koruması: Gürültü ve doku sadece temel statik resimde mevcuttur.
        # Frame başına yeni gürültü eklenmez, böylece videoda karıncalanma (jitter) oluşmaz.
        for i in range(total_frames):
            if motion == "parallax" and work_depth is not None:
                frame = self.apply_3d_parallax(work_img, work_depth, i, total_frames, target_size=(W, H))
            elif motion == "zoom":
                frame = self.apply_zoom(work_img, i, total_frames, target_size=(W, H))
            elif motion == "pan":
                frame = self.apply_pan(work_img, i, total_frames, target_size=(W, H))
            elif motion == "fade":
                frame = self.apply_fade(image, i, total_frames)
            else:
                frame = self.apply_pan(work_img, i, total_frames, target_size=(W, H))

            frames.append(frame)

        return self.encode_frames(frames, fps=fps)
