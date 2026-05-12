import os
import numpy as np
import pickle
from PySide6.QtCore import QThread, Signal
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer
 
 
def converteste_gps(valoare) -> str:
    try:
        grade   = float(valoare[0])
        minute  = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}, {minute:.2f}, {secunde:.2f}"
    except Exception:
        return str(valoare)
 
 
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
#   Ex: Arhitectura are 6 prompturi care acopera: ziua, noaptea,
#   bloc romanesc, strada ploioasa, vedere aeriana etc.
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
 
    # Arhitectura: scene urbane, strazi, blocuri — chiar cu masini parcate
    # Subiectul PRINCIPAL sunt cladirile si spatiul urban, nu vehiculele
    "Arhitectura": [
        "a photo of apartment blocks and residential buildings on a city street",
        "urban street photography showing building facades and sidewalks at night",
        "a Romanian city neighborhood with bloc apartments in winter",
        "a wet city street at night with building lights and reflections on pavement",
        "an aerial or ground level view of a city with buildings and roads",
        "a city skyline or building exterior as the main photographic subject",
    ],
 
    # Vehicule: masina/vehicul ca SUBIECT PRINCIPAL, fotografiat intentionat
    # Imagine tip showroom, portret auto — vehiculul umple cadrul
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
 
 
# ============================================================
# WORKER THREAD
# ============================================================
 
class ScannerWorker(QThread):
    """
    Thread secundar care proceseaza imaginile din fundal.
 
    Pipeline per imagine:
    1. Embedding vizual CLIP (clip-ViT-B-32)
    2. Clasificare prin similitudine cosinus cu centroizii categoriilor
    3. Thumbnail 256x256 in cache
    4. Extragere metadate EXIF (camera, data, GPS)
    5. Salvare SQLite
 
    Semnale:
        progres(curent, total)    → bara de progres
        imagine_reparata(index)   → actualizare live iconita
        finalizat()               → scanare terminata
    """
 
    progres          = Signal(int, int)
    imagine_reparata = Signal(int)
    finalizat        = Signal()
 
    def __init__(self, cale_folder: str, folder_cache: str, cale_db: str, recursiv: bool = False):
        super().__init__()
        self.cale_folder  = cale_folder
        self.folder_cache = folder_cache
        self.cale_db      = cale_db
        self.recursiv     = recursiv
        self.running      = True
        self._model: SentenceTransformer | None = None
 
    def stop(self):
        self.running = False
 
    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate(self.cale_db)
 
        if self._model is None:
            print("[Scanner] Se incarca modelul CLIP...")
            self._model = SentenceTransformer("clip-ViT-B-32")
 
        vectori_categorii = self._pregateste_vectori_categorii()
        fisiere = self._colecteaza_fisiere()
        total = len(fisiere)
 
        if total == 0:
            self.finalizat.emit()
            return
 
        for i, cale_full in enumerate(fisiere):
            if not self.running:
                break
            self._proceseaza_imagine(i, cale_full, db, vectori_categorii, total)
            self.progres.emit(i + 1, total)
 
        self.finalizat.emit()
 
    # ----------------------------------------------------------
    # VECTORI CATEGORII CU ENSEMBLE
    # ----------------------------------------------------------
 
    def _pregateste_vectori_categorii(self) -> np.ndarray:
        """
        Construieste vectorul reprezentativ per categorie.
 
        Categorii cu lista de prompturi → ENSEMBLE:
          - Encodam fiecare prompt separat (normalize_embeddings=True)
          - Calculam media aritmetica → centroid semantic
          - Renormalizam (media vectorilor unitari nu e unitara)
        
        Categorii cu prompt simplu → encoding direct.
 
        Returneaza array (N_categorii x 512) normalizat L2,
        gata pentru dot product cu vectorii imaginilor.
        """
        print("[Scanner] Pregatire vectori categorii (ensemble)...")
        vectori = []
 
        for nume in LISTA_CATEGORII:
            prompt_val = PROMPTS_CLIP[nume]
 
            if isinstance(prompt_val, list):
                v_lista = self._model.encode(
                    prompt_val,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                v_centroid = np.mean(v_lista, axis=0)
                norma = np.linalg.norm(v_centroid)
                if norma > 0:
                    v_centroid = v_centroid / norma
                vectori.append(v_centroid.astype("float32"))
            else:
                v = self._model.encode(
                    [prompt_val],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                vectori.append(v.astype("float32"))
 
        print(f"[Scanner] {len(vectori)} vectori de referinta gata.")
        return np.array(vectori, dtype="float32")
 
    # ----------------------------------------------------------
    # COLECTARE FISIERE
    # ----------------------------------------------------------
 
    def _colecteaza_fisiere(self) -> list[str]:
        formate = (".png", ".jpg", ".jpeg", ".bmp")
        foldere_ignore = {"Windows", "Program Files", "AppData", ".cache"}
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
        total: int,
    ):
        nume_fisier = os.path.basename(cale_full)
 
        existenta = db.cauta_dupa_cale(cale_full)
        if existenta and existenta[10] and os.path.exists(existenta[10]) and existenta[12] is not None:
            return
 
        try:
            with Image.open(cale_full) as img:
                if img.width < 100 or img.height < 100:
                    return
 
                img_fix = ImageOps.exif_transpose(img)
                if img_fix.mode != "RGB":
                    img_fix = img_fix.convert("RGB")
 
                vector_ai  = self._model.encode(img_fix, normalize_embeddings=True)
                categorie  = self._clasifica(vector_ai, vectori_categorii)
                cale_cache = self._salveaza_thumbnail(index, nume_fisier, img_fix)
                date_info  = self._extrage_metadate(
                    img, img_fix, cale_full, cale_cache, vector_ai, categorie
                )
 
                db.salveaza_sau_actualizeaza(date_info)
                self.imagine_reparata.emit(index + 1)
 
        except Exception as e:
            print(f"[Scanner] Eroare la '{nume_fisier}': {e}")
 
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
        nume_cache = f"cache_{index}_{nume_fisier}.png"
        cale_cache = os.path.join(self.folder_cache, nume_cache).replace("\\", "/")
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        thumb.save(cale_cache, "PNG")
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
        date = {
            "cale":       cale_full,
            "nume":       os.path.basename(cale_full),
            "format":     img_original.format,
            "rezolutie":  f"{img_fix.width}x{img_fix.height}",
            "mb":         round(os.path.getsize(cale_full) / (1024 * 1024), 2),
            "cale_cache": cale_cache,
            "vector_ai":  vector_ai,
            "categorie":  categorie,
            "marca":      "Necunoscut",
            "model":      "Necunoscut",
            "data":       "---",
            "gps":        "",
        }
 
        try:
            exif = img_original._getexif()
            if exif:
                for id_tag, val in exif.items():
                    tag = TAGS.get(id_tag, id_tag)
                    if tag == "Make":
                        date["marca"] = str(val).strip()
                    elif tag == "Model":
                        date["model"] = str(val).strip()
                    elif tag == "DateTimeOriginal":
                        date["data"] = str(val)
                    elif tag == "GPSInfo":
                        info_g = {GPSTAGS.get(t, t): val[t] for t in val}
                        if "GPSLatitude" in info_g and "GPSLongitude" in info_g:
                            lat = converteste_gps(info_g["GPSLatitude"])
                            lon = converteste_gps(info_g["GPSLongitude"])
                            date["gps"] = f"Lat: {lat} | Lon: {lon}"
        except Exception:
            pass
 
        return date
    

    def _encode_cu_tta(self, img_fix: Image.Image) -> np.ndarray:
        """Realizeaza 5 tăieri (Centru + Colturi) pentru stabilitate maxima."""
        w, h = img_fix.size
        regiuni = [
            img_fix, # Imaginea intreaga
            img_fix.crop((0, 0, w//2, h//2)),           # Stanga Sus
            img_fix.crop((w//2, 0, w, h//2)),           # Dreapta Sus
            img_fix.crop((0, h//2, w//2, h)),           # Stanga Jos
            img_fix.crop((w//2, h//2, w, h)),           # Dreapta Jos
        ]
        # Encodam cele 5 imagini si calculam media vectorilor (Centroid)
        v_lista = self._model.encode(regiuni, normalize_embeddings=True, show_progress_bar=False) # 
        v_mediu = np.mean(v_lista, axis=0)
        norma = np.linalg.norm(v_mediu)
        return (v_mediu / norma).astype("float32") if norma > 0 else v_mediu

    # INLOCUIESTE metoda existenta in ScannerWorker
    def _proceseaza_imagine(self, index: int, cale_full: str, db, vectori_categorii: np.ndarray, total: int):
        nume_fisier = os.path.basename(cale_full) # 
        existenta = db.cauta_dupa_cale(cale_full) # 
        
        if existenta and existenta[10] and os.path.exists(existenta[10]) and existenta[12] is not None:
            return

        try:
            with Image.open(cale_full) as img:
                if img.width < 100 or img.height < 100: return
                img_fix = ImageOps.exif_transpose(img) # 
                if img_fix.mode != "RGB": img_fix = img_fix.convert("RGB")

                # UPGRADE: Analiza multi-crop in loc de o singura encodare
                vector_ai  = self._encode_cu_tta(img_fix) 
                
                categorie  = self._clasifica(vector_ai, vectori_categorii)
                cale_cache = self._salveaza_thumbnail(index, nume_fisier, img_fix)
                date_info  = self._extrage_metadate(
                    img, img_fix, cale_full, cale_cache, vector_ai, categorie
                )
                db.salveaza_sau_actualizeaza(date_info) # 
                self.imagine_reparata.emit(index + 1) # 
        except Exception as e:
            print(f"[Scanner] Eroare la '{nume_fisier}': {e}")