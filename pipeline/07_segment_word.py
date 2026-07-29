"""
07_segment_word.py — Sağlam glyph segmentasyonu.

Önceki basit sütun-boşluk yöntemi, glyph'in kollarını yanlışlıkla kesebiliyordu
(word1_g1 vakası: 𐰀'nın çapraz kollarından biri kırpma dışında kaldı, model
onu tanıyamadı). Bu script iki temel iyileştirme getiriyor:

1. BAĞLI-BİLEŞEN (connected component) analizi — basit "bu sütunda mürekkep
   var mı" sorgusu yerine, gerçekten birbirine bağlı piksel kümelerini bulur.
   Bir glyph'in kolları birbirinden kopuk görünse bile (bazı fontlarda ince
   çizgiler render sırasında ayrışabilir), yakın bileşenler tek glyph olarak
   BİRLEŞTİRİLİR (x-ekseninde belli bir mesafe içindeyse).

2. CÖMERT KENAR PAYI — glyph'in tespit edilen sınırlarına, en/boyun
   %20-30'u kadar ekstra boşluk eklenir. Dar kırpmak yerine geniş kırpmak
   tercih edilir (fazla boşluk zararsızdır, eksik kırpma glyph'i yok eder).

3. SÖZ AYRACI (":") TESPİTİ — iki nokta üst üste, kendine özgü şekliyle
   (dar, uzun, iki ayrı yuvarlak nokta) otomatik tanınır ve glyph listesinden
   çıkarılıp ayrı işaretlenir (kelime sınırı olarak).

Kullanım:
    python 07_segment_word.py --image kaynak.jpg --out ./data/segmented
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Doğrulama (image (1-3).jpg: 9+7+6=22 glyph) bu değerle üretilmişti — eski
# argparse default'u (25) bu görsellerde tüm glyph'leri tek bileşene
# birleştirip validasyonu bozuyordu, ampirik olarak doğrulanıp düzeltildi.
# Tek doğruluk kaynağı burası; başka yerde (örn. product/app.py) ayrı sabit
# tanımlama, buradan import et.
# TODO: bu sabit piksel değeri, test edilen görsellerin çözünürlüğüne göre
# kalibre edildi. Belirgin farklı çözünürlükte/DPI'da görseller görülürse
# görsel genişliğine oranlı bir değere (örn. img.width * oran) geçmek gerekebilir.
DEFAULT_MERGE_GAP_PX = 6


def find_components(binary_mask, merge_gap_px):
    """
    Bağlı bileşenleri bulur, sonra x-ekseninde merge_gap_px içinde olan
    bileşenleri aynı glyph'e ait sayıp birleştirir (parçalı çizgi/nokta
    sorunlarına karşı).
    """
    labeled, n = ndimage.label(binary_mask)
    if n == 0:
        return []

    objects = ndimage.find_objects(labeled)
    boxes = []
    for sl in objects:
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        boxes.append([x0, y0, x1, y1])

    # x0'a göre sırala, yakın olanları birleştir
    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:
        last = merged[-1]
        gap = b[0] - last[2]
        if gap <= merge_gap_px:
            last[0] = min(last[0], b[0])
            last[1] = min(last[1], b[1])
            last[2] = max(last[2], b[2])
            last[3] = max(last[3], b[3])
        else:
            merged.append(list(b))

    return merged  # [x0, y0, x1, y1] listesi, soldan sağa


# Satır ayrımı için minimum boş-satır (mürekkepsiz satır) yüksekliği.
# Test setindeki tüm tek-satır görsellerde satır içi boş-satır run'ı hiç
# yok (bkz. proje notları); bu yüzden küçük bir eşik bile yanlış bölünmeye
# yol açmadan gerçek çok-satır boşluklarını (11-14px gözlendi) yakalar.
DEFAULT_MIN_LINE_GAP_PX = 4


def split_into_lines(binary_mask, min_gap_px=DEFAULT_MIN_LINE_GAP_PX):
    """
    Yatay projeksiyon (satır başına mürekkep piksel sayısı) kullanarak
    görseli satır bantlarına ayırır. En az min_gap_px ardışık mürekkepsiz
    satır bir satır sınırı sayılır; bundan daha kısa boşluklar (örn. bir
    glyph'in kendi içindeki boşluk) aynı satıra ait sayılıp birleştirilir.

    Döner: [(y0, y1), ...] — üstten alta sıralı, mürekkep içeren satır
    bantları. Tek satırlı bir görselde tek bir bant döner ve bu bant
    find_components'i doğrudan tüm görsel üzerinde çağırmakla birebir
    aynı sonucu verir (üstteki/alttaki boş kenar payı bileşen tespitini
    etkilemez).
    """
    row_has_ink = binary_mask.sum(axis=1) > 0
    h = binary_mask.shape[0]

    bands = []
    y = 0
    while y < h:
        if not row_has_ink[y]:
            y += 1
            continue
        y0 = y
        while y < h and row_has_ink[y]:
            y += 1
        y1 = y

        if bands and (y0 - bands[-1][1]) < min_gap_px:
            bands[-1] = (bands[-1][0], y1)
        else:
            bands.append((y0, y1))

    return bands


def sort_boxes_rtl(line_boxes):
    """
    Bir satırdaki kutuları Göktürkçe okuma sırasına (Sağdan Sola / RTL)
    göre sıralar (x0 değerine göre azalan/büyükten küçüğe).
    """
    return sorted(line_boxes, key=lambda b: b[0], reverse=True)


def find_components_by_line(binary_mask, merge_gap_px, min_line_gap_px=DEFAULT_MIN_LINE_GAP_PX, rtl=True):
    """
    07_segment_word.py'nin asıl giriş noktası: önce satırlara ayırır
    (split_into_lines), sonra HER SATIRI KENDİ İÇİNDE find_components ile
    işler.

    Parametreler:
      - rtl: True ise her satırdaki kutular Göktürkçe okuma yönüne (Sağdan Sola)
        uygun şekilde x0 koordinatına göre azalan sırada sıralanır.
    """
    bands = split_into_lines(binary_mask, min_gap_px=min_line_gap_px)
    if not bands:
        return []

    lines = []
    for y0, y1 in bands:
        sub_mask = binary_mask[y0:y1, :]
        line_boxes = [
            [bx0, by0 + y0, bx1, by1 + y0]
            for bx0, by0, bx1, by1 in find_components(sub_mask, merge_gap_px=merge_gap_px)
        ]
        if rtl:
            line_boxes = sort_boxes_rtl(line_boxes)
        lines.append(line_boxes)

    return lines


def is_colon(box, binary_mask):
    """
    ':' işareti: dar (düşük en, yüksek en/boy oranı) VE İKİ AYRI KARE
    NOKTADAN oluşur — yani kutu içinde y-ekseninde en az bir mürekkepsiz
    ARA BOŞLUK vardır. Sadece dar+uzun aspect oranına bakmak yetmiyor:
    bazı gerçek harfler (örn. dar kanca/zigzag şekilli glyph'ler) da aynı
    aspect aralığına düşebiliyor ama KESİNTİSİZ tek bir çizgiden oluşuyor.
    Tek parça (iç boşluksuz) bir iz asla colon değildir, aspect oranı ne
    olursa olsun.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w == 0 or h == 0:
        return False
    aspect = h / w
    if not (w < 40 and aspect > 1.8):
        return False

    sub = binary_mask[y0:y1, x0:x1]
    row_has_ink = sub.any(axis=1)
    ink_rows = np.where(row_has_ink)[0]
    if len(ink_rows) == 0:
        return False
    first, last = ink_rows[0], ink_rows[-1]
    inner = row_has_ink[first:last + 1]
    return not inner.all()  # aradaki satırlarda boşluk varsa (2 ayrı nokta) True


def crop_with_margin(img: Image.Image, box, left_limit=0, right_limit=None, margin_ratio=0.28, size=256):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin_ratio), int(h * margin_ratio)

    if right_limit is None:
        right_limit = img.width

    x0m = max(left_limit, x0 - mx)
    y0m = max(0, y0 - my)
    x1m = min(right_limit, x1 + mx)
    y1m = min(img.height, y1 + my)

    crop = img.crop((x0m, y0m, x1m, y1m))

    cw, ch = crop.size
    scale = (size * 0.78) / max(cw, ch)  # %78 doluluk, geri kalanı boşluk payı
    new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop_resized = crop.resize((new_w, new_h))

    canvas = Image.new("RGB", (size, size), "white")
    px, py = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(crop_resized, (px, py))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=int, default=128, help="siyah/beyaz eşik (0-255)")
    ap.add_argument("--merge_gap", type=int, default=DEFAULT_MERGE_GAP_PX, help="piksel — bu mesafede olan bileşenler birleştirilir")
    ap.add_argument("--margin_ratio", type=float, default=0.28)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    gray = np.array(img.convert("L"))
    binary = gray < args.threshold

    lines = find_components_by_line(binary, merge_gap_px=args.merge_gap)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    glyph_idx = 0
    colon_count = 0
    manifest = []
    words = []

    # Her satır kendi içinde işlenir: left_limit/right_limit komşuluğu
    # SADECE aynı satırdaki kutulara bakar (bir satırın son glyph'i ile
    # sonraki satırın ilk glyph'i asla komşu sayılmaz). Ayrıca her satır
    # kendi sonunda açık kalan kelimeyi kapatır — satır sonu, kolon
    # olmasa bile örtük bir kelime sınırı sayılır (iki satır aynı
    # kompozisyonda ayrı ifadeler/kelimeler olabilir).
    for line_boxes in lines:
        current = []
        for idx, box in enumerate(line_boxes):
            if is_colon(box, binary):
                colon_count += 1
                manifest.append({"type": "colon", "box": box})
                if current:
                    words.append(current)
                    current = []
                continue

            glyph_idx += 1

            left_limit = 0
            if idx > 0:
                left_limit = (line_boxes[idx - 1][2] + box[0]) // 2

            right_limit = img.width
            if idx < len(line_boxes) - 1:
                right_limit = (box[2] + line_boxes[idx + 1][0]) // 2

            crop = crop_with_margin(
                img, box,
                left_limit=left_limit,
                right_limit=right_limit,
                margin_ratio=args.margin_ratio,
                size=args.size
            )
            fname = f"glyph_{glyph_idx:02d}.png"
            crop.save(out_dir / fname)
            manifest.append({"type": "glyph", "index": glyph_idx, "box": box, "file": fname})
            current.append(fname)

        if current:
            words.append(current)

    print(f"Bulunan satır sayısı: {len(lines)}")
    print(f"Bulunan glyph sayısı: {glyph_idx}")
    print(f"Bulunan söz ayracı (':') sayısı: {colon_count}")
    print(f"Çıktı klasörü: {out_dir}")

    print(f"\nTespit edilen kelime sayısı: {len(words)}")
    for i, w in enumerate(words, 1):
        print(f"  kelime {i}: {w}")


if __name__ == "__main__":
    main()
