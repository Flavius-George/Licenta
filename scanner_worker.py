# ============================================================
# OPTIMIZARI PERFORMANTA (vs varianta initiala)
# ============================================================
# 1. THREAD CPU LIMIT: torch foloseste implicit TOATE nucleele, ceea ce
#    sufoca UI-ul si OS-ul. Limitam la jumatate din nuclee pentru a lasa
#    sistemul responsive in timpul scanarii.
# 2. MODEL PARTAJAT: acceptam SentenceTransformer din main, evitand
#    incarcarea unei a doua copii (~600MB RAM economisit).
# 3. PIL DRAFT MODE pentru JPEG: decodam direct la rezolutie redusa
#    (CLIP foloseste oricum 224x224, nu pierdem semnificatie).
# 4. LIMITA REZOLUTIE PRE-ENCODE: imaginile > 1280px sunt micsorate
#    inainte de encoding → economiseste RAM pentru poze de 24-50MP.
# 5. BATCH WRITE DB: scriem in DB la fiecare 32 imagini → mult mai
#    putine fsync-uri pe disc.
# 6. TTA OPTIONAL (cu_tta=True default): pastreaza acuratetea tezei,
#    dar permite dezactivare pentru scanari rapide (5x mai rapid).
# 7. PROGRES THROTTLED: emitem semnal progres la fiecare 5 imagini sau
#    la fiecare 200ms, nu la fiecare imagine (reduce flood-ul de eventuri).
# ============================================================

import os
import time

# Limitam OpenMP / MKL inainte de import torch/numpy.
# Trebuie setate ca env vars INAINTE de import, nu dupa.
_NUM_THREADS = max(1, (os.cpu_count() or 4) // 2)
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_NUM_THREADS))

import numpy as np
import torch
torch.set_num_threads(_NUM_THREADS)

from PySide6.QtCore import QThread, Signal
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from geocodare import gps_exif_la_decimal, geocodeaza_local
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATIE CATEGORII
# ============================================================

LISTA_CATEGORII = [
    "Oameni", "Mancare", "Animale",
    "Nunti", "Petreceri", "Sarbatori",
    "Documente Clasice", "Diagrame & Scheme", "Baze de Date",
    "Hardware", "Interfete Software", "Screenshots Cod",
    "Natura", "Arhitectura", "Vehicule", "Diverse",
]

# ------------------------------------------------------------------
# PROMPT ENGINEERING PENTRU CLIP — REGULI STRICTE
#
# REGULA #1: ZERO NEGATII ("not", "no", "without", "do NOT")
#   CLIP ignora negatiile! "a car without people" si "a car with people"
#   produc vectori aproape identici. Orice negatie din prompt
#   activeaza tocmai ce vrei sa excluzi.
#
# REGULA #2: ENSEMBLE PENTRU CATEGORII AMBIGUE
#   In loc de 1 prompt per categorie, folosim o LISTA de prompturi.
#   Vectorul final = media normalizata a tuturor vectorilor (centroid).
#   Asta acopera mai multe sub-scenarii si reduce erorile de clasificare.
#
# REGULA #3: SPECIFICITY WINS
#   Descrie atat de specific subiectul incat alte tipuri de imagini
#   sa nu se potriveasca natural — fara sa le mentionezi explicit.
#
# REGULA #4: CUVINTE CHEIE VIZUALE CONCRETE
#   Descrie ce se VEDE fizic: culori, materiale, pozitii, actiuni.
#   CLIP a fost antrenat pe descrieri de imagini de pe internet.
# ------------------------------------------------------------------

PROMPTS_CLIP: dict[str, str | list[str]] = {

    "Oameni": [
        "a portrait photo of a person looking at the camera",
        "a selfie showing a human face up close",
        "a group photo of friends or family smiling together",
        "a candid street photo of people walking in a city",
    ],

    "Mancare": [
        "a food photography shot of a meal on a plate",
        "a close-up of delicious food in a restaurant",
        "a photo of drinks and appetizers on a table",
        "an overhead shot of ingredients and cooking preparation",
    ],

    "Animale": [
        "a close-up portrait of a dog or cat as the main subject",
        "wildlife photography of a wild animal in its habitat",
        "a pet animal photographed indoors sitting or lying down",
        "a bird perched on a branch in sharp focus",
    ],

    "Nunti": [
        "a bride wearing a white wedding dress and veil",
        "wedding ceremony photo with bride groom and guests in a church",
        "a couple exchanging wedding rings at the altar",
        "wedding reception dancing with decorations and wedding cake",
    ],

    "Petreceri": [
        "a birthday party with balloons cake and candles",
        "people laughing and toasting glasses at an indoor celebration",
        "a crowded party with confetti and festive decorations",
        "friends celebrating together with drinks in a party venue",
    ],

    "Sarbatori": [
        "a decorated Christmas tree with lights and ornaments indoors",
        "colorful Easter eggs and spring holiday decorations",
        "fireworks exploding in the night sky over a city",
        "a festive holiday dinner table with candles and decorations",
    ],

    "Documente Clasice": [
        "a scan of a printed paper document with dense text on white background",
        "a photograph of a paper invoice form or official letter",
        "a close-up of handwritten or typed text on paper",
        "a scanned certificate or contract with text and signature fields",
    ],

    "Diagrame & Scheme": [
        "a flowchart with rectangles and arrows on white background",
        "a UML diagram with boxes connected by lines showing workflow",
        "a petri net or state machine diagram with nodes and transitions",
        "a technical schema with geometric shapes and directed arrows",
    ],

    "Baze de Date": [
        "an entity relationship diagram with table boxes and connecting lines",
        "a database schema showing column names and foreign key relations",
        "a relational database ER diagram from MySQL Workbench or DBeaver",
        "a data model diagram with crow foot notation and table structures",
    ],

    "Hardware": [
        "a close-up photo of a computer motherboard with CPU socket",
        "a graphics card or RAM stick isolated on a surface",
        "electronic circuit board components soldered on green PCB",
        "computer hardware parts inside an open PC case",
    ],

    "Interfete Software": [
        "a screenshot of a desktop application with windows and buttons",
        "a mobile phone screen showing an app user interface",
        "a web browser displaying a website with navigation menus",
        "a software dashboard with charts panels and icons on screen",
    ],

    "Screenshots Cod": [
        "a screenshot of Python or JavaScript code with syntax highlighting",
        "a code editor like VS Code showing colored programming code",
        "a terminal window displaying script output or command line code",
        "source code lines with colored keywords in an IDE like PyCharm",
    ],

    "Natura": [
        "a landscape photo of green forest and trees with no buildings",
        "mountain peaks covered in snow under a blue sky",
        "a calm river lake or waterfall surrounded by vegetation",
        "a meadow with wildflowers and open sky at golden hour",
    ],

    "Arhitectura": [
        "a photo of apartment blocks and residential buildings on a city street",
        "urban street photography showing building facades and sidewalks at night",
        "a Romanian city neighborhood with bloc apartments in winter",
        "a wet city street at night with building lights and reflections on pavement",
        "an aerial or ground level view of a city with buildings and roads",
        "a city skyline or building exterior as the main photographic subject",
    ],

    "Vehicule": [
        "automotive photography of a single car filling the entire frame",
        "a car showroom photo with the vehicle as the sole isolated subject",
        "a motorcycle portrait photographed up close on a road or track",
        "an airplane on an airport runway as the main and only subject",
        "a boat or ship on water photographed as the central isolated subject",
    ],

    "Diverse": [
        "an abstract or artistic image with ambiguous subject",
        "a random household object or decorative item photographed closely",
        "a blurry unfocused image with unclear content",
        "a heavily edited stylized image with unusual colors",
    ],
}

PRAG_CLASIFICARE = 0.20

# Limita rezolutie inainte de CLIP. CLIP face oricum resize la 224x224,
# deci nu pierdem informatie utila. Reduce dramatic RAM-ul pentru poze
# moderne (24-50MP).
DIM_MAX_PRE_ENCODE = 1280

# Numar de imagini procesate inainte de un flush in DB
BATCH_DB = 32

# Indecsi coloane DB pentru verificarea cache-ului la re-scanare
_COL_CACHE_SCANNER  = 15
_COL_VECTOR_SCANNER = 17


# ============================================================
# WORKER THREAD
# ============================================================

class ScannerWorker(QThread):
    """
    Thread secundar care proceseaza imaginile din fundal.

    Pipeline per imagine:
    1. Embedding vizual CLIP (clip-ViT-B-32), optional cu TTA (5 crop-uri)
    2. Clasificare prin similitudine cosinus cu centroizii categoriilor
    3. Thumbnail 256x256 in cache
    4. Extragere metadate EXIF (camera, data, GPS)
    5. Geocodare inversa locala (lat/lon → oras, tara)
    6. Salvare SQLite (in batch-uri)

    Semnale:
        progres(curent, total)    → bara de progres
        imagine_reparata(index)   → actualizare live iconita
        finalizat()               → scanare terminata
    """

    progres          = Signal(int, int)
    imagine_reparata = Signal(int)
    finalizat        = Signal()

    def __init__(
        self,
        cale_folder: str,
        folder_cache: str,
        cale_db: str,
        recursiv: bool = False,
        model: SentenceTransformer | None = None,
        cu_tta: bool = True,
    ):
        super().__init__()
        self.cale_folder  = cale_folder
        self.folder_cache = folder_cache
        self.cale_db      = cale_db
        self.recursiv     = recursiv
        self.running      = True
        # Modelul poate fi pasat din main pentru a evita o a doua incarcare in RAM
        self._model: SentenceTransformer | None = model
        # Daca este False, scanarea este de ~5x mai rapida (un singur embedding)
        self._cu_tta = cu_tta
        # Buffer pentru scrieri in batch
        self._buffer_db: list[dict] = []
        self._ultimul_progres_emis = 0.0

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate(self.cale_db)

        t_start = time.monotonic()

        # ---- PASUL 1: Colectam fisierele FARA a incarca modelul ----
        fisiere = self._colecteaza_fisiere()
        if not fisiere:
            print("[Scanner] Niciun fisier de procesat in folder.")
            self.finalizat.emit()
            return

        # ---- PASUL 2: Filtram imaginile DEJA procesate ----
        # Astfel evitam complet incarcarea modelului CLIP daca nu e nimic
        # nou de procesat (rescan rapid).
        t0 = time.monotonic()
        deja_procesate = db.obtine_set_complet_procesate()
        fisiere_noi = [c for c in fisiere if c not in deja_procesate]
        nr_skip = len(fisiere) - len(fisiere_noi)
        print(f"[Scanner] {len(fisiere)} fisiere gasite, {nr_skip} deja procesate, "
              f"{len(fisiere_noi)} de procesat (filtrare {time.monotonic()-t0:.2f}s)")

        if not fisiere_noi:
            print("[Scanner] Nimic nou de facut - skip total!")
            self.finalizat.emit()
            return

        # ---- PASUL 3: Acum incarcam modelul (doar daca avem ce procesa) ----
        if self._model is None:
            print("[Scanner] Se incarca modelul CLIP (nu a fost partajat din main)...")
            t0 = time.monotonic()
            self._model = SentenceTransformer("clip-ViT-B-32")
            print(f"[Scanner] Model incarcat in {time.monotonic()-t0:.1f}s")
        else:
            print("[Scanner] Folosim modelul CLIP partajat din main (economie RAM).")

        # Info despre device-ul folosit (CPU vs CUDA)
        try:
            device = next(self._model._first_module().parameters()).device
            print(f"[Scanner] Inferenta CLIP pe: {device}")
        except Exception:
            pass

        # ---- PASUL 4: Vectorii de categorii (cu cache pe disc) ----
        t0 = time.monotonic()
        vectori_categorii = self._incarca_sau_calc_vectori_categorii()
        print(f"[Scanner] Vectori categorii gata in {time.monotonic()-t0:.2f}s")

        # ---- PASUL 5: Scanarea propriu-zisa ----
        total = len(fisiere_noi)
        print(f"[Scanner] Pornim procesarea: {total} imagini, "
              f"TTA={'ON' if self._cu_tta else 'OFF'}, threads CPU={_NUM_THREADS}")

        t_scan = time.monotonic()
        for i, cale_full in enumerate(fisiere_noi):
            if not self.running:
                break
            self._proceseaza_imagine(i, cale_full, db, vectori_categorii)

            if len(self._buffer_db) >= BATCH_DB:
                self._flush_buffer(db)

            acum = time.monotonic()
            if (i + 1) % 5 == 0 or (acum - self._ultimul_progres_emis) > 0.2 or i + 1 == total:
                self.progres.emit(i + 1, total)
                self._ultimul_progres_emis = acum

        self._flush_buffer(db)

        # ---- Raport final timing ----
        scan_elapsed = time.monotonic() - t_scan
        total_elapsed = time.monotonic() - t_start
        if total > 0:
            print(f"[Scanner] Scanare incheiata in {scan_elapsed:.1f}s "
                  f"({scan_elapsed/total:.2f}s per imagine)")
        print(f"[Scanner] TIMP TOTAL (cu overhead): {total_elapsed:.1f}s")

        self.finalizat.emit()

    # ----------------------------------------------------------
    # BATCH WRITE
    # ----------------------------------------------------------

    def _flush_buffer(self, db):
        """Scrie tot batch-ul curent in DB intr-o singura tranzactie."""
        if not self._buffer_db:
            return
        try:
            db.salveaza_batch(self._buffer_db)
        except Exception as e:
            print(f"[Scanner] Eroare la batch insert: {e}")
            # Fallback: incercam unul cate unul ca sa nu pierdem totul
            for d in self._buffer_db:
                try:
                    db.salveaza_sau_actualizeaza(d)
                except Exception as e2:
                    print(f"[Scanner] Eroare individuala: {e2}")
        self._buffer_db.clear()

    # ----------------------------------------------------------
    # VECTORI CATEGORII (cu cache pe disc + single-batch encoding)
    # ----------------------------------------------------------

    def _incarca_sau_calc_vectori_categorii(self) -> np.ndarray:
        """
        Incarca vectorii categoriilor din cache pe disc daca exista,
        altfel ii calculeaza si ii salveaza. Hash-ul prompturilor
        asigura ca un edit in PROMPTS_CLIP invalideaza automat cache-ul.
        """
        import hashlib
        import json
        import pickle

        # Cheie unica bazata pe continutul prompturilor + lista categoriilor
        continut = json.dumps(
            {"prompts": PROMPTS_CLIP, "lista": LISTA_CATEGORII},
            sort_keys=True,
        ).encode()
        hash_prompts = hashlib.md5(continut).hexdigest()[:12]
        cache_file = os.path.join(
            self.folder_cache, f"vectori_categorii_{hash_prompts}.pkl"
        ).replace("\\", "/")

        # Cache HIT: incarcam din pickle (rapid, sub 100ms)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    vectori = pickle.load(f)
                print(f"[Scanner] Vectori categorii incarcati din cache.")
                return vectori
            except Exception as e:
                print(f"[Scanner] Cache invalid, recalculam: {e}")

        # Cache MISS: calculam o singura data si salvam
        vectori = self._calc_vectori_categorii_bulk()
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(vectori, f)
            print("[Scanner] Vectori categorii salvati in cache pentru rulari viitoare.")
        except Exception as e:
            print(f"[Scanner] Nu am putut salva cache vectori: {e}")
        return vectori

    def _calc_vectori_categorii_bulk(self) -> np.ndarray:
        """
        OPTIMIZARE: encodeaza TOATE prompturile (din toate categoriile)
        intr-un SINGUR forward pass (batch mare), apoi calculeaza
        centroizii. Inainte erau 16 forward pass-uri separate.

        Returneaza array (N_categorii x 512) normalizat L2,
        gata pentru dot product cu vectorii imaginilor.
        """
        print("[Scanner] Calculam vectorii de categorii (single-batch)...")

        # Adunam toate prompturile intr-o singura lista, retinem cate
        # apartin fiecarei categorii
        all_prompts: list[str] = []
        contoare: list[int] = []
        for nume in LISTA_CATEGORII:
            prompt_val = PROMPTS_CLIP[nume]
            if isinstance(prompt_val, list):
                all_prompts.extend(prompt_val)
                contoare.append(len(prompt_val))
            else:
                all_prompts.append(prompt_val)
                contoare.append(1)

        # Un singur forward pass, batch mare → mult mai rapid
        all_vectors = self._model.encode(
            all_prompts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )

        # Calculam centroidul per categorie (ensemble) si renormalizam
        vectori = []
        idx = 0
        for n in contoare:
            if n == 1:
                vectori.append(all_vectors[idx].astype("float32"))
            else:
                centroid = np.mean(all_vectors[idx:idx + n], axis=0)
                norma = np.linalg.norm(centroid)
                if norma > 0:
                    centroid = centroid / norma
                vectori.append(centroid.astype("float32"))
            idx += n

        print(f"[Scanner] {len(vectori)} vectori de referinta calculati "
              f"din {len(all_prompts)} prompt-uri.")
        return np.array(vectori, dtype="float32")

    # ----------------------------------------------------------
    # COLECTARE FISIERE
    # ----------------------------------------------------------

    def _colecteaza_fisiere(self) -> list[str]:
        formate = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        foldere_ignore = {"Windows", "Program Files", "AppData", ".cache",
                          "$RECYCLE.BIN", "System Volume Information",
                          "node_modules", ".git"}
        fisiere = []

        try:
            if self.recursiv:
                for radacina, directoare, nume_fisiere in os.walk(self.cale_folder):
                    directoare[:] = [d for d in directoare if d not in foldere_ignore]
                    for nume in nume_fisiere:
                        if nume.lower().endswith(formate):
                            fisiere.append(os.path.join(radacina, nume).replace("\\", "/"))
            else:
                for nume in os.listdir(self.cale_folder):
                    cale = os.path.join(self.cale_folder, nume).replace("\\", "/")
                    if os.path.isfile(cale) and nume.lower().endswith(formate):
                        fisiere.append(cale)
        except Exception as e:
            print(f"[Scanner] Eroare la cautare fisiere: {e}")

        return sorted(fisiere)

    # ----------------------------------------------------------
    # PROCESARE IMAGINE
    # ----------------------------------------------------------

    def _proceseaza_imagine(
        self,
        index: int,
        cale_full: str,
        db,
        vectori_categorii: np.ndarray,
    ):
        """
        Proceseaza o imagine: embedding, clasificare, thumbnail, EXIF,
        geocodare. Adauga rezultatul in self._buffer_db pentru scriere
        in batch ulterioara.
        Sareste imaginea daca thumbnail-ul si vectorul exista deja in DB.
        """
        nume_fisier = os.path.basename(cale_full)
        existenta = db.cauta_dupa_cale(cale_full)

        # Skip daca avem deja vector + thumbnail valid
        if (existenta
                and existenta[_COL_CACHE_SCANNER]
                and os.path.exists(existenta[_COL_CACHE_SCANNER])
                and existenta[_COL_VECTOR_SCANNER] is not None):
            return

        try:
            with Image.open(cale_full) as img:
                # ---- OPTIMIZARE PIL DRAFT: pentru JPEG, decodam direct
                # la rezolutie redusa. Mult mai rapid si folosim mai putin RAM.
                if img.format == "JPEG":
                    img.draft("RGB", (DIM_MAX_PRE_ENCODE, DIM_MAX_PRE_ENCODE))

                img.load()  # forteaza decode (Image.open este lazy)

                if img.width < 100 or img.height < 100:
                    return

                img_fix = ImageOps.exif_transpose(img)
                if img_fix.mode != "RGB":
                    img_fix = img_fix.convert("RGB")

                # ---- LIMITA REZOLUTIE PRE-ENCODE
                # CLIP face resize intern la 224x224. Nu are sens sa
                # incarcam o poza de 6000x4000 in RAM pentru asta.
                if max(img_fix.size) > DIM_MAX_PRE_ENCODE:
                    img_fix.thumbnail(
                        (DIM_MAX_PRE_ENCODE, DIM_MAX_PRE_ENCODE),
                        Image.LANCZOS,
                    )

                # Embedding (TTA optional)
                vector_ai  = self._encode_imagine(img_fix)
                categorie  = self._clasifica(vector_ai, vectori_categorii)
                cale_cache = self._salveaza_thumbnail(index, nume_fisier, img_fix)
                date_info  = self._extrage_metadate(
                    img, img_fix, cale_full, cale_cache, vector_ai, categorie
                )

                # Punem in buffer, nu scriem inca in DB
                self._buffer_db.append(date_info)
                self.imagine_reparata.emit(index + 1)

        except Exception as e:
            print(f"[Scanner] Eroare la '{nume_fisier}': {e}")

    def _encode_imagine(self, img_fix: Image.Image) -> np.ndarray:
        """Routes la encode normal sau TTA in functie de flag."""
        if self._cu_tta:
            return self._encode_cu_tta(img_fix)
        # Encoding rapid: doar imaginea intreaga (~5x mai rapid decat TTA)
        v = self._model.encode(
            [img_fix],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return v.astype("float32")

    def _encode_cu_tta(self, img_fix: Image.Image) -> np.ndarray:
        """
        Test-Time Augmentation: 5 crop-uri (imaginea intreaga + 4 colturi).
        Vectorul final = centroid normalizat al celor 5 embedding-uri.
        Reduce erorile de clasificare pentru imagini cu subiecte excentrice.
        Toate cele 5 crop-uri sunt encodate intr-un singur batch CLIP.
        """
        w, h = img_fix.size
        regiuni = [
            img_fix,                                       # Imaginea intreaga
            img_fix.crop((0,      0,      w // 2, h // 2)),  # Stanga Sus
            img_fix.crop((w // 2, 0,      w,      h // 2)),  # Dreapta Sus
            img_fix.crop((0,      h // 2, w // 2, h)),       # Stanga Jos
            img_fix.crop((w // 2, h // 2, w,      h)),       # Dreapta Jos
        ]
        v_lista = self._model.encode(regiuni, normalize_embeddings=True, show_progress_bar=False)
        v_mediu = np.mean(v_lista, axis=0)
        norma = np.linalg.norm(v_mediu)
        return (v_mediu / norma).astype("float32") if norma > 0 else v_mediu.astype("float32")

    def _clasifica(self, vector_ai: np.ndarray, vectori_categorii: np.ndarray) -> str:
        """
        Dot product intre vectorul imaginii (normalizat) si centroizii
        categoriilor (normalizati) = similitudine cosinus.
        """
        scoruri = np.dot(vectori_categorii, vector_ai)
        idx_max = int(np.argmax(scoruri))
        if scoruri[idx_max] > PRAG_CLASIFICARE:
            return LISTA_CATEGORII[idx_max]
        return "Diverse"

    def _salveaza_thumbnail(self, index: int, nume_fisier: str, img: Image.Image) -> str:
        import hashlib
        # Hash al numelui fisierului pentru a evita coliziunile intre scanari
        hash_nume = hashlib.md5(nume_fisier.encode()).hexdigest()[:8]
        nume_cache = f"cache_{index}_{hash_nume}_{nume_fisier}.png"
        cale_cache = os.path.join(self.folder_cache, nume_cache).replace("\\", "/")
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        # compress_level=3 e un bun compromis (default e 6, mai lent)
        thumb.save(cale_cache, "PNG", compress_level=3)
        return cale_cache

    def _extrage_metadate(
        self,
        img_original: Image.Image,
        img_fix: Image.Image,
        cale_full: str,
        cale_cache: str,
        vector_ai: np.ndarray,
        categorie: str,
    ) -> dict:
        """
        Extrage metadate EXIF, rezolvând problemele specifice iPhone (GPS IFD)
        și asigurând extragerea datei originale a fotografiei.
        """
        date = {
            "cale":      cale_full,
            "nume":      os.path.basename(cale_full),
            "format":    img_original.format,
            "rezolutie": f"{img_fix.width}x{img_fix.height}",
            "mb":        round(os.path.getsize(cale_full) / (1024 * 1024), 2),
            "cale_cache": cale_cache,
            "vector_ai":  vector_ai,
            "categorie":  categorie,
            "marca":      "Necunoscut",
            "model":      "Necunoscut",
            "data":       "---",
            "gps":        "",
            "lat":        None,
            "lon":        None,
            "oras":       "",
            "tara":       "",
            "tara_cod":   "",
        }

        try:
            exif_data = img_original.getexif()
            if exif_data:
                # 1. Metadate Generale (Marca, Model)
                for tag_id, val in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "Make":
                        date["marca"] = str(val).strip()
                    elif tag == "Model":
                        date["model"] = str(val).strip()
                    elif tag == "DateTime":
                        # Fallback dacă nu găsim DateTimeOriginal mai jos
                        if date["data"] == "---":
                            date["data"] = str(val)

                # 2. Extragere Dată Originală (Capture Time) - IFD 0x8769
                exif_ifd = exif_data.get_ifd(0x8769)
                if exif_ifd:
                    data_orig = exif_ifd.get(0x9003) or exif_ifd.get(0x9004)
                    if data_orig:
                        date["data"] = str(data_orig)

                # 3. Extragere GPS (IFD 0x8825 - Esențial pentru iPhone)
                gps_ifd = exif_data.get_ifd(0x8825)
                if gps_ifd:
                    gps_raw = {GPSTAGS.get(t, t): gps_ifd[t] for t in gps_ifd}

                    if "GPSLatitude" in gps_raw and "GPSLongitude" in gps_raw:
                        lat = gps_exif_la_decimal(
                            gps_raw["GPSLatitude"],
                            gps_raw.get("GPSLatitudeRef", "N")
                        )
                        lon = gps_exif_la_decimal(
                            gps_raw["GPSLongitude"],
                            gps_raw.get("GPSLongitudeRef", "E")
                        )

                        if lat is not None and lon is not None:
                            date["lat"] = lat
                            date["lon"] = lon
                            date["gps"] = f"Lat: {lat:.5f} | Lon: {lon:.5f}"

                            # Geocodare locală (Țară/Oraș)
                            loc = geocodeaza_local(lat, lon)
                            date["oras"] = loc["oras"]
                            date["tara"] = loc["tara"]
                            date["tara_cod"] = loc["cod"]

        except Exception as e:
            print(f"[Scanner] Atentie: Nu am putut extrage toate metadatele pentru {date['nume']}: {e}")

        return date
