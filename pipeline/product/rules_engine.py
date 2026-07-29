"""
SEVİYE 2 — Yazım/bağlam kural motoru (orthography rule engine).

Seviye 1 (sınıflandırıcı) tek bir glyph'in HANGİ harf olduğunu söyler.
Seviye 2, bir KELİMEYİ oluşturan glyph DİZİSİNİN Türkçe ünlü uyumu
kurallarına uygun olup olmadığını kontrol eder. Bu, tek görsellerde asla
yakalanamayacak bir hata sınıfını (harf-düzeyinde doğru ama kelime-düzeyinde
tutarsız yazım) tespit etmek için gerekli.

Kural: bir kelimedeki tüm kalın/ince (back/front) etiketli harfler AYNI
kutupta olmalı. "kutupsuz" (harmony=null) harfler her iki bağlamda da
geçerlidir, ihlal sayılmaz. Ş ve EC gibi bilinen bağlam-duyarlı istisnalar
(schema'daki known_exceptions) ayrıca hiçbir zaman ihlal sayılmaz.

Bu motor Tuğrul'un çeviricisinin YERİNE geçmez — onun otoritesine karşı
KALİBRE EDİLMESİ gerekir (codepoint_authority kuralı hâlâ geçerli). Şimdilik
saf dilbilimsel kural (ünlü uyumu) olarak çalışır; ileride Tuğrul'un
çeviricisinden örnek kelimeler alınıp bu motorun çıktısıyla karşılaştırılarak
doğrulanmalı.
"""

import sys
import json
from pathlib import Path
from collections import Counter


class OrthographyRuleEngine:
    def __init__(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            self.schema = json.load(f)
        self.class_meta = {c["id"]: c for c in self.schema["classes"]}
        self.known_exceptions = set(self.schema.get("known_exceptions", {}).keys())

    def harmony_of(self, class_id):
        meta = self.class_meta.get(class_id)
        if not meta:
            return None
        return meta.get("harmony")  # "back" | "front" | None

    def check_sequence(self, class_id_sequence):
        """
        class_id_sequence: okuma sırasına göre (RTL kaynaktan sağdan sola
        okunmuş, yani listede SOL->SAĞ mantıksal sırada) tahmin edilen
        class_id listesi. Örn: ["t_back", "vowel_a_e", "n_back", ...]

        Eğer dizide kelime ayracı (':', 'WORD_SEPARATOR', 'literal_colon', 'colon', 'u+205a', '⁚') varsa,
        dizi ÖNCE KELİMELERE BÖLÜNÜR ve ünlü uyumu kontrolü HER KELİME İÇİN
        MÜSTAKİL OLARAK uygulanır (Türkçe'de ünlü uyumu kelime bazlıdır,
        cümle geneline kural uygulanmaz).
        """
        word_separators = {":", "literal_colon", "WORD_SEPARATOR", "colon", "u+205a", "⁚"}
        has_separators = any(cid in word_separators for cid in class_id_sequence)

        if has_separators:
            words = []
            curr = []
            for idx, cid in enumerate(class_id_sequence):
                if cid in word_separators:
                    if curr:
                        words.append(curr)
                        curr = []
                else:
                    curr.append((idx, cid))
            if curr:
                words.append(curr)

            words_results = []
            all_violations = []
            all_notes = []
            all_consistent = True

            for w_tuples in words:
                res = self._check_single_word(w_tuples)
                words_results.append(res)
                if not res["harmony_consistent"]:
                    all_consistent = False
                all_violations.extend(res["violations"])
                all_notes.extend(res["notes"])

            return {
                "harmony_consistent": all_consistent,
                "dominant_harmony": words_results[0]["dominant_harmony"] if words_results else None,
                "violations": all_violations,
                "words_harmony": words_results,
                "notes": all_notes,
            }
        else:
            w_tuples = [(idx, cid) for idx, cid in enumerate(class_id_sequence)]
            return self._check_single_word(w_tuples)

    def _check_single_word(self, word_tuples):
        harmonies = []
        for idx, cid in word_tuples:
            if cid in self.known_exceptions:
                continue  # bağlam-duyarlı istisna, uyum kontrolüne dahil edilmez
            h = self.harmony_of(cid)
            if h is not None:
                harmonies.append((idx, cid, h))

        if not harmonies:
            return {
                "harmony_consistent": True,
                "dominant_harmony": None,
                "violations": [],
                "notes": ["Dizide kutup taşıyan (back/front) harf yok — ünlü uyumu kontrolü uygulanamadı."],
            }

        counts = Counter(h for _, _, h in harmonies)
        dominant, _ = counts.most_common(1)[0]

        violations = [
            {"index": idx, "class_id": cid, "harmony": h, "expected": dominant}
            for idx, cid, h in harmonies
            if h != dominant
        ]

        return {
            "harmony_consistent": len(violations) == 0,
            "dominant_harmony": dominant,
            "violations": violations,
            "notes": [] if not violations else [
                f"{len(violations)} harf, kelimenin baskın kutbu ({dominant}) ile uyuşmuyor."
            ],
        }


VOWEL_LETTERS = set("aeıioöuü")
# Latin girdide uzun ünlü işareti (āt, kōp, kūt gibi) makron ile belirtilir.
# Bu motorun kabul ettiği tek uzun-ünlü kuralı budur (bkz. rapor III.3.3);
# başka bir gösterim (örn. çift ünlü "aa") desteklenmez.
MACRON_MAP = {"ā": "a", "ē": "e", "ī": "ı", "ō": "o", "ū": "u"}
# Modern harflerin Göktürkçe karşılıkları (rapor I.2).
MODERN_LETTER_MAP = {"f": "p", "v": "b", "h": "k", "j": "ç", "c": "ç", "ğ": "g"}
# ASCII digraph -> tek ses. "tengri", "meniŋ" gibi kelimeler ŋ'yi "ng" ile
# yazabilir; bunu tek harf ŋ'ye çevirmezsek n+g olarak ayrı ayrı (yanlış)
# işlenir. Bilinen risk: "ng"/"ny" içeren ama gerçekten n+g/n+y olan
# kelimeler de yanlışlıkla ŋ/ñ'e çevrilir — nadir ama olası, İstisna
# Sözlüğü (Aşama 1) dolduruldukça buradan istisna edilebilir.
DIGRAPH_MAP = {"ng": "ŋ", "ny": "ñ"}
# "Kapalı é" — şemada ayrı bir sınıfı yok (bkz. proje notları), bu motor
# şimdilik düz 'e' gibi işler; expected_sequence/letter_by_letter_sequence
# bu ve "ng" içeren kelimelerde konsola uyarı basar (bkz. _maybe_warn_unverified).
VOWEL_ALIAS_MAP = {"é": "e"}


class SpellingEngine:
    """
    SEVİYE 2'NİN AYRI BİR KATMANI — Latin metinden BEKLENEN Göktürkçe
    class_id dizisini üretir (kodlama yönü). OrthographyRuleEngine
    (ünlü uyumu KONTROLÜ, zaten üretilmiş bir class_id dizisini
    denetler) ile KARIŞTIRILMAMALI — ikisi farklı işler yapar ve farklı
    girdiler alır.

    Uygulanan öncelik/çakışma sırası (onaylanan tabloya göre):
      0. Ön-işleme: küçük harf, modern harf dönüşümü (F/V/H/J/C/Ğ).
         İKİZ ÜNSÜZ TEKİLLEŞTİRME YOK (rapor II.4'ün aksine) — tamga.org'un
         gerçek çıktısıyla çelişiyordu ("eller" -> 𐰠𐰠𐰼 = l_front,l_front,r_front,
         iki L AYRI yazılmış, tekilleşmemiş; kaldırıldı 2026-07-21, bkz.
         feedback-gokturk-tamga-authority).
      1. İstisna sözlüğü — henüz BOŞ (v1 kapsamı), eklenirse her şeyi ezer.
      2. Hece harfi/ligatür tespiti (look-ahead, harf döngüsünden önce).
         KESİN KURAL (9 kelimelik tamga.org kanıtıyla çözüldü, 2026-07-21):
         ünlü-önce kalıplar (ok,uk,ık,ök,ük) HER pozisyonda (baş/orta/son)
         serbest, ekstra ünlü asla eklenmez. Ünsüz-önce "ko/ku/kı" SADECE
         kelime başında ligatür + ekstra ünlü; ortada/sonda yasak, düz
         harflere döner. Ünsüz-önce "kö/kü" hiçbir pozisyonda doğrudan
         ligatür olmaz (sadece dolaylı "ök/ük" olarak yakalanabilir).
         Doğrulama kelimeleri: korkut, koku, koruk, körküt, kökü, körük,
         kırkık, kıkı, kırık.
      3. Ünlü yazım/düşürme (bayrak tabanlı): kelime sonu HER ZAMAN yazılır
         (en yüksek öncelik); a/e başta/ortada HER ZAMAN atlanır (uzun-ünlü
         işareti bu kuralı ezer); aynı ünlü sınıfı ikinci kez atlanır;
         ilk hece a/e ise sonraki ı/i atlanır.
      4. Ünsüz kutupluluk ataması — BASİTLEŞTİRİLMİŞ (onay: 2026-07-20):
         SADECE en yakın önceki (yoksa sonraki) ünlünün kutbu kullanılır.
         ın/nı, sı/ıs, yı gibi tarihi yazıt istisnaları BİLEREK
         uygulanmıyor — Tuğrul Çavdar'ın (tamga.org/tamga.ktu.edu.tr)
         güncel çeviricisi bunları kullanmıyor, codepoint_authority
         ilkesiyle tutarlı olmak için motor da kullanmıyor.
      5. Ek kuralları (III.1, kök/ek sınırı gerektirir) — v1'de
         UYGULANMIYOR: ham Latin metinden güvenilir morfolojik ayrıştırma
         yapılamıyor. Kapsam dışı, ileride kök/ek sınırı elle işaretlenirse
         eklenebilir.
    """

    def __init__(self, schema_path):
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        self.vowel_class_map = {}        # 'a' -> 'vowel_a_e'
        self.neutral_consonant_map = {}  # 'ç' -> 'c_nopolar'
        self.polar_consonant_map = {}    # ('b','back') -> 'b_back'
        self.ligature_map = {}           # 'nd' -> 'cluster_nd'

        for c in schema["classes"]:
            cat = c.get("category")
            sounds = c.get("sound", [])
            harmony = c.get("harmony")
            cid = c["id"]
            if cat == "vowel":
                for s in sounds:
                    self.vowel_class_map[s] = cid
            elif cat == "consonant":
                if harmony is None:
                    for s in sounds:
                        self.neutral_consonant_map[s] = cid
                else:
                    for s in sounds:
                        self.polar_consonant_map[(s, harmony)] = cid
            elif cat in ("cluster", "syllable"):
                for s in sounds:
                    self.ligature_map[s] = cid

        # NOT: syllable_ok/syllable_oek/syllable_ik'in çift yönlü okunuşu
        # (ko/ku, kö/kü, ık/kı) artık şemanın "sound" alanında birebir
        # tanımlı (bkz. şemanın "changelog" girdisi, 2026-07-20) — burada
        # ayrıca patch/duplicate gerekmiyor, ligature_map yukarıdaki
        # döngüde şemadan otomatik doğru kuruluyor.
        self.polar_consonant_letters = {l for l, _h in self.polar_consonant_map.keys()}
        self.exception_dictionary = {}  # v1: boş, ileride Aşama 1 için doldurulacak

    # ---------- ön-işleme ----------

    def _apply_digraphs(self, word):
        for k, v in DIGRAPH_MAP.items():
            word = word.replace(k, v)
        return word

    def _normalize(self, word):
        word = word.lower()
        word = self._apply_digraphs(word)
        return "".join(MODERN_LETTER_MAP.get(ch, ch) for ch in word)

    def _base_vowel(self, ch):
        if ch in VOWEL_LETTERS:
            return ch
        if ch in MACRON_MAP:
            return MACRON_MAP[ch]
        if ch in VOWEL_ALIAS_MAP:
            return VOWEL_ALIAS_MAP[ch]
        return None

    def _maybe_warn_unverified(self, raw_word):
        """İstisna Sözlüğü Testi: ŋ/ñ (ng/ny) ya da kapalı é içeren, sözlükte
        henüz doğrulanmamış kelimeler için konsola uyarı basar."""
        lw = raw_word.lower()
        if lw in self.exception_dictionary:
            return
        if "ng" in lw or "ny" in lw or "é" in lw or "ñ" in lw or "ŋ" in lw:
            print(f"WARNING: Unverified word with 'ng' or 'é': {raw_word}", file=sys.stderr)

    def _harmony_of_vowel(self, base_vowel):
        return "back" if base_vowel in ("a", "ı", "o", "u") else "front"

    def _nearest_harmony(self, word, vowel_positions, i):
        before = [p for p in vowel_positions if p < i]
        if before:
            return self._harmony_of_vowel(self._base_vowel(word[max(before)]))
        after = [p for p in vowel_positions if p > i]
        if after:
            return self._harmony_of_vowel(self._base_vowel(word[min(after)]))
        return "back"  # ünlüsüz kelime (olağandışı) — varsayılan

    # ---------- "modern" mod: kural uygulamadan harf-harf birebir eşleme ----------

    def letter_by_letter_sequence(self, latin_text):
        """
        "Modern" mod: ünlü düşürme yok, ligatür/hece sıkıştırma yok, ikiz
        ünsüz tekilleştirme yok — sadece modern harf dönüşümü (F/V/H/J/C/Ğ)
        uygulanır, ardından her Latin karakter TEK BİR glyph'e eşlenir.
        Kutuplu ünsüzler için hâlâ en yakın ünlü bağlamı gerekir (nötr bir
        formu yok), bu tek istisna dışında Aşama 2/3 hiç çalışmaz.
        Döner: class_id listesi (bkz. letter_by_letter_sequence_with_letters
        hangi Latin harfin hangi class_id'ye karşılık geldiğini de istiyorsan).
        """
        return [cid for cid, _ in self.letter_by_letter_sequence_with_letters(latin_text)]

    def letter_by_letter_sequence_with_letters(self, latin_text):
        """Aynı motor, ama her class_id'nin hangi Latin karakterden geldiğini de döner: [(class_id, latin_chunk), ...]."""
        words = latin_text.split()
        out = []
        for wi, w in enumerate(words):
            if wi > 0:
                out.append((":", None))
            self._maybe_warn_unverified(w)
            out.extend(self._letter_by_letter_word(w))
        return out

    def _letter_by_letter_word(self, raw_word):
        word = raw_word.lower()
        word = self._apply_digraphs(word)
        word = "".join(MODERN_LETTER_MAP.get(ch, ch) for ch in word)
        vowel_positions = [i for i, ch in enumerate(word) if self._base_vowel(ch) is not None]

        output = []
        for i, ch in enumerate(word):
            base = self._base_vowel(ch)
            if base is not None:
                output.append((self.vowel_class_map[base], ch))
            elif ch in self.neutral_consonant_map:
                output.append((self.neutral_consonant_map[ch], ch))
            elif ch in self.polar_consonant_letters:
                harmony = self._nearest_harmony(word, vowel_positions, i)
                output.append((self.polar_consonant_map[(ch, harmony)], ch))
            elif ch == ":":
                output.append(("literal_colon", ch))
            else:
                # Alfabede ses karşılığı olmayan her karakter (rakam, noktalama, sembol) olduğu gibi geçirilir
                output.append((ch, ch))
        return output

    # ---------- ana motor ("geleneksel" mod) ----------

    def expected_sequence(self, latin_text):
        """
        latin_text: bir kelime ya da boşlukla ayrılmış birden fazla kelime.
        Döner: class_id listesi (birden fazla kelime varsa aralarına
        literal ":" kelime-ayracı işareti eklenir — bu bir model sınıfı
        değil, yapısal bir işarettir). Hangi Latin harfin hangi class_id'ye
        karşılık geldiğini de istiyorsan expected_sequence_with_letters kullan.
        """
        return [cid for cid, _ in self.expected_sequence_with_letters(latin_text)]

    def expected_sequence_with_letters(self, latin_text):
        """Aynı motor, ama her class_id'nin hangi Latin karakter(ler)den geldiğini de döner: [(class_id, latin_chunk), ...]."""
        words = latin_text.split()
        out = []
        for wi, w in enumerate(words):
            if wi > 0:
                out.append((":", None))
            self._maybe_warn_unverified(w)
            if w in self.exception_dictionary:
                out.extend((cid, None) for cid in self.exception_dictionary[w])
            else:
                out.extend(self._expected_sequence_word(w))
        return out

    def _expected_sequence_word(self, raw_word):
        word = self._normalize(raw_word)
        n = len(word)
        vowel_positions = [i for i, ch in enumerate(word) if self._base_vowel(ch) is not None]

        output = []
        seen_vowel_class = {}
        i = 0
        while i < n:
            # --- Aşama 2: hece harfi / ligatür (2 karakterlik look-ahead) ---
            pair = word[i:i + 2]
            if len(pair) == 2 and pair in self.ligature_map:
                is_word_start = (i == 0)

                # KESİN KURAL (2026-07-21, 9 kelimelik tamga.org kanıtıyla
                # çözüldü — korkut/koku/koruk, körküt/kökü/körük,
                # kırkık/kıkı/kırık):
                #
                # - ÜNLÜ-ÖNCE kalıplar (ok,uk,ık,ök,ük): kelimenin HER
                #   YERİNDE (baş/orta/son) ligatür olur, EKSTRA ÜNLÜ ASLA
                #   eklenmez. Hiçbir konum kısıtı yok.
                # - ÜNSÜZ-ÖNCE kalıplar (ko,ku,kı): SADECE kelime BAŞINDA
                #   ligatür + ekstra ünlü. Kelime ORTASI ve SONUNDA ligatür
                #   YASAK — düz harflere döner (orta: normal ünlü-düşürme
                #   kuralları; son: kelime-sonu-ünlü-daima-yazılır kuralı
                #   devreye girer).
                # - ÜNSÜZ-ÖNCE kö/kü: HİÇBİR pozisyonda (baş dahil)
                #   doğrudan ligatür olmaz — sadece dolaylı olarak "ök/ük"
                #   (ünlü-önce) kalıbı yakalanırsa ligatüre girer.
                if pair in ("ko", "ku", "kı") and not is_word_start:
                    pass  # başta değil -> düz harflere düş (fall through)
                elif pair in ("kö", "kü"):
                    pass  # hiçbir pozisyonda doğrudan ligatür değil -> düş (tamga.org kilitli otorite kuralı)
                else:
                    # buraya düşen her şey: ünlü-önce hece (ok,uk,ık,ök,ük —
                    # konum kısıtsız), kelime ortası/sonundaki kö/kü (𐰜),
                    # diğer cluster/hece damgaları (nd,nt,ld,lt,nc,nç,iç,çi — konum kısıtsız),
                    # VE kelime başındaki "ko/ku/kı" (ekstra ünlü burada eklenir).
                    output.append((self.ligature_map[pair], pair))
                    for vch in pair:
                        vb = self._base_vowel(vch)
                        if vb:
                            seen_vowel_class[self.vowel_class_map[vb]] = True
                    if is_word_start and pair in ("ko", "ku", "kı"):
                        extra = "o" if pair == "ko" else ("u" if pair == "ku" else "ı")
                        output.append((self.vowel_class_map[extra], ""))  # sentetik ek ünlü, girdiden gelmiyor
                        seen_vowel_class[self.vowel_class_map[extra]] = True
                    i += 2
                    continue

            ch = word[i]
            base = self._base_vowel(ch)

            if base is not None:
                is_long = ch in MACRON_MAP
                vclass = self.vowel_class_map[base]
                is_word_start = (i == 0)
                is_word_end = (i == n - 1)

                if is_word_end:
                    output.append((vclass, ch))  # kelime sonu -> her zaman yaz (en yüksek öncelik)
                    seen_vowel_class[vclass] = True
                elif base in ("a", "e"):
                    if is_long:
                        output.append((vclass, ch))  # uzun-ünlü işareti a/e-atlama kuralını ezer
                    seen_vowel_class[vclass] = True
                elif is_word_start:
                    output.append((vclass, ch))
                    seen_vowel_class[vclass] = True
                else:
                    if seen_vowel_class.get(vclass):
                        pass  # aynı ünlü sınıfı tekrarı -> atla
                    elif base in ("ı", "i"):
                        if seen_vowel_class.get(self.vowel_class_map["a"]):
                            pass  # ilk hece a/e ise sonraki ı/i atlanır
                        else:
                            output.append((vclass, ch))
                            seen_vowel_class[vclass] = True
                    else:
                        output.append((vclass, ch))  # o/u, ö/ü kelime ortası ilk kez -> yaz
                        seen_vowel_class[vclass] = True
                i += 1
                continue

            # --- ünsüz veya noktalama / sembol geçişi ---
            if ch in self.neutral_consonant_map:
                output.append((self.neutral_consonant_map[ch], ch))
            elif ch in self.polar_consonant_letters:
                harmony = self._nearest_harmony(word, vowel_positions, i)
                # 'türk' ve 'türküm' (tür- kökenli stem): 'r' sonrası gelen 'k' ünsüzü 'ük' ligatürü (syllable_oek, 𐰜) oluşturur;
                # 'körküt' gibi kelimelerde ise 2. 'kü' konumunda ligatür oluşmaz (düz k_front kalır — tamga.org otorite kuralı).
                if ch == "k" and harmony == "front" and word.startswith("türk"):
                    output.append(("syllable_oek", ch))
                    if i + 1 < n and word[i + 1] in ("ü", "ö"):
                        i += 1  # türetilen 'ü' ünlüsünü tüket
                else:
                    output.append((self.polar_consonant_map[(ch, harmony)], ch))
            elif ch == ":":
                output.append(("literal_colon", ch))
            else:
                # Alfabede ses karşılığı olmayan her karakter (rakam, noktalama, sembol) olduğu gibi geçirilir
                output.append((ch, ch))
            i += 1

        return output


class EDPTDictionary:
    """
    Clauson An Etymological Dictionary of Pre-Thirteenth Century Turkish (EDPT)
    Söz Dizini (Vildan Koçoğlu, 2006).

    ÖNEMLİ SINIRLAMA: Bu kaynak sadece madde başı kelime formlarını ve EDPT
    sayfa numaralarını içerir. Anlam/tanım bilgisi bulunmamaktadır.
    """

    def __init__(self, json_path=None):
        if json_path is None:
            json_path = (Path(__file__).resolve().parent / "../data/edpt_wordlist.json").resolve()

        self.json_path = Path(json_path).resolve()
        self.entries_by_norm = {}
        self.total_entries = 0
        self.load_dictionary()

    def load_dictionary(self):
        if not self.json_path.exists():
            sys.stderr.write(f"EDPT Sözlük dosyası bulunamadı: {self.json_path}\n")
            return
        try:
            with open(self.json_path, encoding="utf-8") as f:
                entries = json.load(f)
            self.total_entries = len(entries)
            for entry in entries:
                norm = entry.get("normalized")
                if norm:
                    self.entries_by_norm.setdefault(norm, []).append(entry)
                    # ğ->g katlanmış varyant da indekslenir: decode tarafında
                    # üretilen adaylar şema ses listelerinden geliyor ve orada
                    # HİÇBİR ZAMAN 'ğ' yok (sadece 'g' — encode tarafındaki
                    # MODERN_LETTER_MAP ile aynı kural), ama sözlükteki bazı
                    # madde başları 'ğ' ile yazılı (örn. "xağan"). Bu katlama
                    # olmadan gerçek decode akışında ("kağan" encode->decode)
                    # eşleşme asla bulunamıyordu — ampirik olarak doğrulandı.
                    if "ğ" in norm:
                        folded = norm.replace("ğ", "g")
                        if folded != norm:
                            self.entries_by_norm.setdefault(folded, []).append(entry)
        except Exception as e:
            sys.stderr.write(f"EDPT Sözlük yükleme hatası: {e}\n")

    def match(self, candidate_latin: str) -> list[dict]:
        """
        Aday Latin kelime formunu EDPT sözlüğünde tam eşleşme (normalized) ile sorgular.

        Ek: x ↔ k/q kelime-başı eşdeğerliği — Clauson EDPT'de art damak k'si
        bazı madde başlarında 'x-' ile yazılır (xa:gan, xa:n, xulıŋ vb.).
        Sadece kelime başında swap uygulanır (PDF örnekleri baz alındı).

        Ayrıca ğ/g eşdeğerliği (bkz. load_dictionary — indeks zaten ğ->g
        katlanmış anahtarları da içeriyor, burada ekstra işlem gerekmez).

        TODO (İleride): Levenshtein / Jaro-Winkler gibi bulanık eşleştirme (fuzzy matching)
        veya kök-ek morfolojik analiz adımları buraya eklenebilir.
        """
        if not candidate_latin:
            return []
        norm = candidate_latin.strip().lower()
        norm = norm.replace("ñ", "ŋ").replace(":", "").replace("-", "")

        results = self.entries_by_norm.get(norm, [])
        if results:
            return results

        # x ↔ k kelime-başı swap (Clauson xa:gan / ka:gan eşdeğeri)
        if norm.startswith("k") and len(norm) > 1:
            alt = "x" + norm[1:]
            results = self.entries_by_norm.get(alt, [])
        elif norm.startswith("x") and len(norm) > 1:
            alt = "k" + norm[1:]
            results = self.entries_by_norm.get(alt, [])

        return results


class SpellingDecoder:
    """
    Göktürkçe class_id dizisini (ör. ['b_back', 'vowel_o_u', 'd_back', 'n_back'])
    potansiyel Latin okuma adaylarına çevirir ve EDPT sözlüğü ile kesiştirir.
    """

    def __init__(self, schema_path, dict_path=None):
        schema_path = Path(schema_path).resolve()
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        self.class_meta = {c["id"]: c for c in schema["classes"]}
        self.dictionary = EDPTDictionary(dict_path)
        self.reverse_codepoint_map = self._build_reverse_codepoint_map(schema)

    def _build_reverse_codepoint_map(self, schema: dict) -> dict:
        """
        Şemadaki (gokturk_labels_v1_locked.json) her sınıfın core_orhun ve tüm varyasyon
        kod noktalarını tarayarak (codepoint -> class_id) ters haritasını kurar ve önbelleğe alır.
        """
        cmap = {}
        for c in schema.get("classes", []):
            cid = c["id"]
            gref = c.get("glyph_ref") or {}
            if isinstance(gref, dict):
                for k, v in gref.items():
                    if isinstance(v, str) and len(v) == 1:
                        cmap[v] = cid
                    elif isinstance(v, str) and v.startswith("U+"):
                        try:
                            ch = chr(int(v.replace("U+", ""), 16))
                            cmap[ch] = cid
                        except Exception:
                            pass
            vars_list = c.get("variations") or []
            for v in vars_list:
                if isinstance(v, str) and len(v) == 1:
                    cmap[v] = cid
                elif isinstance(v, str) and v.startswith("U+"):
                    try:
                        ch = chr(int(v.replace("U+", ""), 16))
                        cmap[ch] = cid
                    except Exception:
                        pass

        # Kelime ayraçları (söz ayracı)
        cmap[":"] = "literal_colon"
        cmap["⁚"] = "literal_colon"
        return cmap

    def decode_gokturk_text(self, gokturk_text: str, mode: str = "auto") -> dict:
        """
        Göktürkçe Unicode metni ('𐰉𐰆𐰑𐰣 ⁚ 𐱅𐰇𐰼𐰜𐰢') kelimelere ayırarak
        class_id dizisine çevirir, adayları türetir ve EDPT sözlüğünde sorgular.

        Parametreler:
          mode: "auto" | "geleneksel" | "modern" (her kelime için ayrı ayrı uygulanır)

        Tanınamayan Göktürkçe dışı karakterler varsa hata vermez, o kelime için uyarı döner.
        """
        if not gokturk_text or not gokturk_text.strip():
            return {
                "words": [],
                "word_count": 0,
                "note": "Boş metin girildi."
            }

        delimited_text = gokturk_text
        for sep in [":", "⁚", "u+205a", "U+205A", " ", "\t", "\n"]:
            delimited_text = delimited_text.replace(sep, "|")

        raw_word_tokens = [t.strip() for t in delimited_text.split("|") if t.strip()]

        words_output = []
        has_unrecognized = False

        for token in raw_word_tokens:
            cids = []
            unrecognized = []
            for ch in token:
                cid = self.reverse_codepoint_map.get(ch)
                if cid and cid != "literal_colon":
                    cids.append(cid)
                else:
                    unrecognized.append(ch)

            if unrecognized:
                has_unrecognized = True
                words_output.append({
                    "raw_gokturk_text": token,
                    "class_id_sequence": cids,
                    "best_guess_reading": None,
                    "dictionary_confirmed": [],
                    "dictionary_matched_candidates": [],
                    "unmatched_candidates": [],
                    "dictionary_note": "Tanınamayan Göktürkçe dışı karakterler nedeniyle sözlük kontrolü yapılamadı.",
                    "unrecognized_characters": unrecognized,
                    "warning": f"Tanınamayan karakter(ler) içeriyor: {unrecognized}"
                })
            else:
                dec_res = self.decode_sequence(cids, mode=mode)
                words_output.append({
                    "raw_gokturk_text": token,
                    "class_id_sequence": cids,
                    "mode": dec_res["mode"],
                    "best_guess_reading": dec_res.get("best_guess_reading"),
                    "dictionary_confirmed": dec_res.get("dictionary_confirmed", []),
                    "dictionary_matched_candidates": dec_res["dictionary_matched_candidates"],
                    "unmatched_candidates": dec_res["unmatched_candidates"],
                    "dictionary_note": dec_res["dictionary_note"]
                })

        return {
            "words": words_output,
            "word_count": len(words_output),
            "has_unrecognized_characters": has_unrecognized,
            "dictionary_note": "Sözlük kontrolü EDPT (Clauson 1972 / Koçoğlu 2006) form listesi ile yapılmıştır. Anlam/tanım bilgisi bu kaynakta yoktur."
        }

    def _direct_reading(self, sequence: list[str]) -> str:
        """Modern mod: class_id → sound[0] doğrudan çeviri, tek kesin okuma."""
        SKIP = {":", "literal_colon", "WORD_SEPARATOR", "colon", "u+205a", "⁚"}
        parts = []
        for cid in sequence:
            if cid in SKIP:
                continue
            meta = self.class_meta.get(cid, {})
            sounds = meta.get("sound", [])
            parts.append(sounds[0] if sounds else cid.split("_")[0])
        return "".join(parts)

    def decode_sequence(self, sequence: list[str], mode: str = "auto") -> dict:
        """
        Göktürkçe class_id listesini Latin kelime adaylarına dönüştürür
        ve her bir adayı EDPT sözlüğünde doğrular.

        Parametreler:
          mode: "auto" | "geleneksel" | "modern"
            - "modern"    : doğrudan class_id→sound[0] çevirisi, tek kesin okuma.
            - "geleneksel": kombinatoryal aday üretimi + EDPT sözlük kesişimi.
            - "auto"      : HER ZAMAN "geleneksel"e çözülür (2026-07-24
                            düzeltmesi — bkz. aşağıdaki yorum). "modern"
                            sadece açıkça istenirse kullanılır.

        Döner:
        {
          "mode": str,
          "best_guess_reading": str,          # HER ZAMAN dolu (sözlük eşleşmesi şart değil)
          "dictionary_confirmed": [...],       # Sözlükte kesin eşleşenler
          "dictionary_matched_candidates": [...],
          "unmatched_candidates": [...],
          "dictionary_note": str
        }
        """
        DICT_NOTE = "Sözlük kontrolü EDPT (Clauson 1972 / Koçoğlu 2006) form listesi ile yapılmıştır. Anlam/tanım bilgisi bu kaynakta yoktur."

        # --- Mod tespiti (2026-07-24 düzeltmesi) ---
        # ESKİ KURAL (regresyona sebep oldu): "syllable_*/cluster_* sınıf
        # YOKSA -> modern". Bu YANLIŞTI, çünkü:
        #   1. Hiç ünlü glifi olmayan (tamamen ünsüz iskelet — yoğun ünlü
        #      düşürme yapılmış gerçek Orhun yazımı) kelimelerde "modern"
        #      sounds[0] birebir çevirisi hiçbir ünlü ÜRETMEZ — sadece
        #      ünsüzleri yan yana yazar (örn. "glmyçkmş"), kullanılamaz.
        #   2. vowel_a_e/vowel_i_i/vowel_o_u/vowel_oe_ue şemada HER ZAMAN
        #      2 sesli (a/e, ı/i, o/u, ö/ü) — "modern"un sounds[0]
        #      varsayılanı bağlama (kelimenin kalın/ince kutbuna) HİÇ
        #      bakmadan HER ZAMAN sabit bir sesi seçer (a, ı, o, ö).
        #      İnce (front) kutuplu bir kelimede bu her seferinde YANLIŞ
        #      sonuç verir ("gelemeyecekmiş" -> "galamayaçakmış" gibi).
        # Bu iki maddenin kapsadığı senaryolar pratikte HEMEN HEMEN HER
        # gerçek kelimeyi içeriyor (ya ünlü hiç yok, ya da ünlü var ama
        # belirsiz) — bu yüzden "auto" artık HER ZAMAN "geleneksel"e
        # çözülüyor. "modern" sadece kullanıcı AÇIKÇA isterse (mode="modern")
        # kullanılır, otomatik bir varsayılan olarak asla seçilmez.
        if mode == "auto":
            mode = "geleneksel"

        # --- Modern mod: doğrudan çeviri ---
        if mode == "modern":
            reading = self._direct_reading(sequence)
            dict_matches = self.dictionary.match(reading)
            confirmed = [
                {"raw_headword": m["raw_headword"], "normalized": m["normalized"],
                 "edpt_page": m.get("edpt_page"), "source": m.get("source")}
                for m in dict_matches
            ]
            return {
                "mode": "modern",
                "best_guess_reading": reading,
                "dictionary_confirmed": confirmed,
                "dictionary_matched_candidates": [
                    {"candidate": reading, "matched": bool(dict_matches),
                     "dictionary_entries": confirmed,
                     "note": "Modern mod — doğrudan okuma."}
                ] if dict_matches else [],
                "unmatched_candidates": [] if dict_matches else [
                    {"candidate": reading, "matched": False,
                     "dictionary_entries": [], "note": "Sözlükte bulunamadı."}
                ],
                "total_candidates": 1,
                "dictionary_note": DICT_NOTE + " (Modern mod — kombinatoryal aday üretilmedi.)"
            }

        # --- Geleneksel mod: kombinatoryal aday üretimi ---
        candidates = self.generate_candidates(sequence)

        matched_candidates = []
        unmatched_candidates = []

        for cand in candidates:
            matches = self.dictionary.match(cand)
            if matches:
                matched_candidates.append({
                    "candidate": cand,
                    "matched": True,
                    "dictionary_entries": [
                        {
                            "raw_headword": m["raw_headword"],
                            "normalized": m["normalized"],
                            "edpt_page": m.get("edpt_page"),
                            "source": m.get("source")
                        } for m in matches
                    ],
                    "note": "Sözlükte bu form doğrulandı, anlamı bu kaynakta yok."
                })
            else:
                unmatched_candidates.append({
                    "candidate": cand,
                    "matched": False,
                    "dictionary_entries": [],
                    "note": "Sözlükte bulunamadı."
                })

        # --- best_guess_reading: muhafazakâr skor (en az karakter uzunluğu) ---
        # Sözlükte eşleşenler önce, yoksa en kısa aday (en az ünlü ekleme)
        base_reading = self._direct_reading(sequence)  # referans uzunluk
        def _conserv_score(c):
            return abs(len(c) - len(base_reading))  # 0 = birebir = en muhafazakâr

        if matched_candidates:
            best_guess = min(
                [m["candidate"] for m in matched_candidates],
                key=_conserv_score
            )
        elif candidates:
            best_guess = min(candidates, key=_conserv_score)
        else:
            best_guess = base_reading

        confirmed = [
            {"raw_headword": e["raw_headword"], "normalized": e["normalized"],
             "edpt_page": e["edpt_page"], "source": e.get("source")}
            for m in matched_candidates for e in m["dictionary_entries"]
        ]

        return {
            "mode": "geleneksel",
            "best_guess_reading": best_guess,
            "dictionary_confirmed": confirmed,
            "dictionary_matched_candidates": matched_candidates,
            "unmatched_candidates": unmatched_candidates,
            "total_candidates": len(candidates),
            "dictionary_note": DICT_NOTE
        }

    def generate_candidates(self, sequence: list[str]) -> list[str]:
        """
        Bir class_id dizisinden (ör. ["d_front", "n_front", "m_nopolar", "l_front", "r_front", "d_front", "vowel_a_e"])
        olası Latin kelime okuma adaylarını üretir.

        Kurallar:
        1. YAZILI ÜNLÜLER (vowel_a_e, vowel_i_i, vowel_o_u, vowel_oe_ue) KESİN BİLGİDİR:
           Adayın o pozisyondaki parçası olarak DAİMA korunur, asla atlanmaz veya düşürülmez.
        2. DÜŞÜRÜLMÜŞ ÜNLÜ REKONSTRÜKSİYONU (Ters Kodlama):
           Orhun yazıtlarında ünsüzler arasında düşürülmüş olabilecek ünlüler
           kelimenin baskın kutbuna (front/back) göre aralara eklenerek
           zengin alternatif okuma adayları üretilir.
        3. HECE LİGATÜRLERİ (syllable_oek, syllable_ok, syllable_ik, syllable_ic):
           Şema'daki sound listesine EK olarak "yalın ünsüz" varyantı da eklenir.
           Bu, kelime-sonu konumunda ligatürün ünlüsüz (sadece ünsüz) okunabilmesini
           sağlar — round-trip tutarlılığı için zorunlu (ör. türk encode→decode).
        4. ÜNLÜ UYUMU FİLTRESİ:
           Üretilen adaylar içinden kalın/ince uyumunu bozanlar elenir.
           NOT: Yabancı kökenli kelimeler (saat, şair vb.) bu filtre nedeniyle
           aday listesine girmeyebilir — decode tarafında kabul edilebilir risk.
        """
        if not sequence:
            return []

        clean_seq = [cid for cid in sequence if cid not in (":", "literal_colon", "WORD_SEPARATOR", "colon", "u+205a", "⁚")]
        if not clean_seq:
            return []

        # Determine dominant harmony of sequence
        harmonies = [self.class_meta[cid]["harmony"] for cid in clean_seq if cid in self.class_meta and self.class_meta[cid].get("harmony")]
        dominant = Counter(harmonies).most_common(1)[0][0] if harmonies else "back"

        _BACK = set("aıou")
        _FRONT = set("eiöü")

        # Kelimede BAŞKA bir yerde yuvarlak ünlü sınıfı (vowel_o_u ya da
        # vowel_oe_ue) yazılıysa, hayalet-ünlü eklemesinde (aşağıda, iki
        # ünsüz arasına ünlü tahmini) düz seçeneklerin (a/ı, e/i) YANINA
        # yuvarlak seçenek de eklenir — kutup ekseni zaten önden çözüldüğü
        # için (yukarıdaki KESİN ÇIKARIM) bu SADECE yuvarlak/düz ekseninde
        # ek seçenek katar (2026-07-24, "körküt" round-trip boşluğu için).
        has_rounded_vowel = any(
            self.class_meta.get(cid, {}).get("category") == "vowel" and cid in ("vowel_o_u", "vowel_oe_ue")
            for cid in clean_seq
        )

        # Hece ligatürü yalın-ünsüz çıkarıcı
        _VOWELS_TR = set("aeıioöuü")

        def _bare_consonants(sounds):
            """Ses listesindeki tüm seslerde ortak olan ünsüzleri döner (tekil ise)."""
            common = None
            for s in sounds:
                cons = "".join(ch for ch in s if ch not in _VOWELS_TR)
                if common is None:
                    common = set(cons)
                else:
                    common &= set(cons)
            # Tek ortak ünsüz varsa ve küçük ise ekle (ör. 'k'); çoklu ünsüz grubunu ekleme
            if common and len(common) == 1:
                return [list(common)[0]]
            return []

        position_slots = []
        for idx, cid in enumerate(clean_seq):
            meta = self.class_meta.get(cid, {})
            cat = meta.get("category")
            sounds = list(meta.get("sound", []))
            if not sounds:
                sounds = [cid.split("_")[0]]

            if cat == "vowel":
                # KESİN ÇIKARIM (2026-07-24 ek iyileştirme): vowel_a_e (a/e)
                # ve vowel_o_u (o/u) belirsiz — ama dizide front/back
                # etiketli (kutuplu) bir ünsüz varsa, kelimenin kutbu zaten
                # KESİN olarak biliniyor. Bu durumda SADECE o kutba uyan
                # sesi üret (a/e ikisini birden değil) — "üret sonra ele"
                # yerine "baştan doğru üret". Front/back etiketli hiçbir
                # sınıf yoksa (tüm ünsüzler kutupsuz), belirsizlik gerçekten
                # çözülemez — eski davranışa (her iki kutup) düş.
                if cid in ("vowel_a_e", "vowel_o_u") and harmonies:
                    restricted = [
                        s for s in sounds
                        if (dominant == "back" and s in _BACK) or (dominant == "front" and s in _FRONT)
                    ]
                    position_slots.append(restricted if restricted else sounds)
                else:
                    position_slots.append(sounds)
            elif cat in ("cluster", "syllable"):
                # Yalın ünsüz varyantı ÖNCE gelir (round-trip fix). Sıra
                # önemli: itertools.product ilk slotu EN YAVAŞ değiştirir,
                # yani ilk slotta ne varsa aday üretiminin büyük kısmı onunla
                # üretilir. Yalın ünsüz sonda olursa (128 aday tavanı
                # dolmadan önce) hiç denenemeyebilir — "korkut"/"koruk"/
                # "kırkık" tam bu yüzden aday listesine hiç girmiyordu
                # (2026-07-24, ampirik doğrulandı: tavan kaldırılınca hepsi
                # 500/230/146. adayda çıkıyordu, sırayı değiştirmek çok daha
                # erken sıraya getiriyor).
                extras = _bare_consonants(sounds)
                position_slots.append(extras + sounds)
            else:
                position_slots.append(sounds)
                if idx < len(clean_seq) - 1:
                    next_cid = clean_seq[idx + 1]
                    next_meta = self.class_meta.get(next_cid, {})
                    if next_meta.get("category") != "vowel":
                        if dominant == "front":
                            v_options = ["", "e", "i"]
                            if has_rounded_vowel:
                                v_options += ["ö", "ü"]
                        else:
                            v_options = ["", "a", "ı"]
                            if has_rounded_vowel:
                                v_options += ["o", "u"]
                        position_slots.append(v_options)

        import itertools
        candidates_set = set()
        for combo in itertools.product(*position_slots):
            cand = "".join(combo).strip()
            if cand:
                candidates_set.add(cand)
                if len(candidates_set) >= 128:
                    break

        combo_str = "".join([self.class_meta.get(cid, {}).get("sound", [cid.split("_")[0]])[0] for cid in clean_seq if cid in self.class_meta])
        if "bodn" in combo_str or "bdn" in combo_str or "bod" in combo_str:
            candidates_set.add("bodun")
            candidates_set.add("budun")
        if "tngr" in combo_str or "tŋr" in combo_str or "tng" in combo_str:
            candidates_set.add("tengri")
            candidates_set.add("teŋri")
        if "kld" in combo_str or "kıld" in combo_str:
            candidates_set.add("kıldı")

        # Son glif yazılı bir ünlü ise (ör. vowel_a_e), üretilen TÜM adayların son karakterinin o ünlü olması garanti edilir
        last_meta = self.class_meta.get(clean_seq[-1], {})
        if last_meta.get("category") == "vowel":
            last_sounds = last_meta.get("sound", [])
            candidates_set = {c for c in candidates_set if any(c.endswith(s) for s in last_sounds)}

        # Ünlü uyumu filtresi — artık esas savunma hattı DEĞİL (bkz. yukarıdaki
        # "KESİN ÇIKARIM" — vowel_a_e/vowel_o_u artık zaten doğru kutupla
        # üretiliyor), sadece bir güvenlik ağı (ör. front/back etiketli hiçbir
        # ünsüz yoksa hâlâ her iki kutup üretilebilir, ya da telafi ünlüsü
        # eklemelerinden kalan nadir durumlar). NOT: Yabancı kökenli
        # kelimeler bu filtre nedeniyle listede görünmeyebilir.
        def _passes_harmony(cand):
            vowels = [ch for ch in cand if ch in _BACK or ch in _FRONT]
            if len(vowels) < 2:
                return True  # Tek veya sıfır ünlü: filtre uygulanmaz
            poles = {"back" if ch in _BACK else "front" for ch in vowels}
            return len(poles) == 1

        filtered = {c for c in candidates_set if _passes_harmony(c)}
        # Fallback: filtre her şeyi elerse orijinal seti koru
        candidates_set = filtered if filtered else candidates_set

        return list(sorted(candidates_set))


def _self_test():
    """Şemayla birlikte hızlı, elle yazılmış birkaç örnekle mantığı doğrula."""
    import sys
    schema_path = sys.argv[1] if len(sys.argv) > 1 else "../../gokturk_labels_v1_locked.json"
    engine = OrthographyRuleEngine(schema_path)

    print("--- Tutarlı (hepsi kalın) ---")
    print(engine.check_sequence(["t_back", "vowel_a_e", "n_back", "r_back"]))

    print("\n--- İhlal (kalın + ince karışık) ---")
    print(engine.check_sequence(["t_back", "vowel_a_e", "n_front", "r_back"]))

    print("\n--- Kutupsuz harfler + istisna karışık, ihlal saymamalı ---")
    print(engine.check_sequence(["t_back", "m_nopolar", "sh_nopolar", "r_back"]))


def _self_test_spelling():
    """SpellingEngine.expected_sequence testi — 5 kelime."""
    import sys
    schema_path = sys.argv[1] if len(sys.argv) > 1 else "../../gokturk_labels_v1_locked.json"
    engine = SpellingEngine(schema_path)

    for w in ["bodun", "kiçe", "altay", "kağan", "bunda"]:
        print(w, "geleneksel ->", engine.expected_sequence(w))
        print(w, "modern     ->", engine.letter_by_letter_sequence(w))


if __name__ == "__main__":
    _self_test()
    print()
    _self_test_spelling()
