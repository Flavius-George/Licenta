import os
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer

# Functie pentru coordonate GPS (fara diacritice)
def converteste_gps(valoare):
    try:
        grade = float(valoare[0])
        minute = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}, {minute:.2f}, {secunde:.2f}"
    except:
        return str(valoare)

class ScannerWorker(QThread):
    progres = Signal(int, int)      # Pentru Status Bar (numara tot)
    imagine_reparata = Signal(int)  # Se emite cand procesam o imagine noua (cache + AI)
    finalizat = Signal()

    def __init__(self, cale_folder):
        super().__init__()
        self.cale_folder = cale_folder
        self.running = True
        # Incarcam modelul o singura data la initializare
        # ViT-B-32 este echilibrul perfect intre viteza si acuratete
        self.model = None 

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate()
        
        # 1. PREGATIRE MODEL AI
        # Incarcam modelul in interiorul thread-ului pentru a nu bloca interfata
        if self.model is None:
            self.model = SentenceTransformer('clip-ViT-B-32')

        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        try:
            # SORTARE ALFABETICA pentru sincronizare cu UI
            nume_fisiere = os.listdir(self.cale_folder)
            fisiere = sorted([f for f in nume_fisiere if f.lower().endswith(formate)])
        except:
            return

        total = len(fisiere)
        folder_cache = os.path.join(os.getcwd(), ".cache")
        if not os.path.exists(folder_cache):
            os.makedirs(folder_cache)

        for i, nume in enumerate(fisiere):
            if not self.running:
                break
            
            current = i + 1
            cale_full = os.path.join(self.cale_folder, nume)
            
            # Verificam ce avem deja in DB
            existenta = db.cauta_dupa_cale(cale_full)

            # LOGICA DE SKIP INTELIGENT:
            # existenta[10] = cale_cache, existenta[11] = vector_ai
            are_cache = existenta and len(existenta) > 10 and existenta[10] and os.path.exists(existenta[10])
            are_vector = existenta and len(existenta) > 11 and existenta[11] is not None

            if are_cache and are_vector:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    # 1. CORECTIE ORIENTARE
                    img_fix = ImageOps.exif_transpose(img)
                    
                    # 2. GENERARE VECTOR AI (CLIP)
                    # Folosim imaginea "dreapta" pentru ca AI-ul sa vada corect
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    
                    # 3. GENERARE CACHE (Thumbnail)
                    nume_cache = f"cache_{nume}.png"
                    cale_cache = os.path.join(folder_cache, nume_cache)
                    
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((1024, 1024)) 
                    img_thumb.save(cale_cache, "PNG")

                    # 4. COLECTARE DATE
                    date_info = {
                        'cale': cale_full,
                        'nume': nume,
                        'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache,
                        'vector_ai': vector_ai # Adaugam vectorul pentru DB
                    }
                    
                    # 5. EXIF
                    exif_raw = img._getexif()
                    if exif_raw:
                        for id_tag, valoare in exif_raw.items():
                            n_tag = TAGS.get(id_tag, id_tag)
                            if n_tag == "Make": date_info['marca'] = valoare
                            if n_tag == "Model": date_info['model'] = valoare
                            if n_tag == "DateTimeOriginal": date_info['data'] = valoare
                            if n_tag == "GPSInfo":
                                info_g = {}
                                for t in valoare:
                                    s_tag = GPSTAGS.get(t, t)
                                    info_g[s_tag] = valoare[t]
                                if "GPSLatitude" in info_g:
                                    lat = converteste_gps(info_g["GPSLatitude"])
                                    lon = converteste_gps(info_g["GPSLongitude"])
                                    date_info['gps'] = f"Lat: {lat} | Lon: {lon}"

                    # 6. SALVARE IN DB
                    db.salveaza_sau_actualizeaza(date_info)
                    
                    # Anuntam UI-ul ca avem o imagine "gata" (corectata + AI)
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare procesare AI/Cache pentru {nume}: {e}")

            self.progres.emit(current, total)

        self.finalizat.emit()