import os
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer

# Functie pentru coordonate GPS 
def converteste_gps(valoare):
    try:
        grade = float(valoare[0])
        minute = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}, {minute:.2f}, {secunde:.2f}"
    except:
        return str(valoare)

class ScannerWorker(QThread):
    progres = Signal(int, int)      
    imagine_reparata = Signal(int)  
    finalizat = Signal()

    def __init__(self, cale_folder):
        super().__init__()
        self.cale_folder = cale_folder
        self.running = True
        self.model = None 
        
        # --- DEFINIRE CATEGORII SMART ---
        self.categorii_config = {
            "Oameni": "a photo of a person, a portrait, or a group of people",
            "Natura": "a natural landscape, mountains, forest, trees, or a beach",
            "Tehnologie": "electronic devices, computer hardware, gadgets, or circuit boards",
            "Documente": "a screenshot, a document with text, a scan of a paper, or a book",
            "Arhitectura": "buildings, city streets, houses, or architecture",
            "Vehicule": "cars, motorcycles, airplanes, or transport vehicles",
            "Animale": "a photo of a pet, a dog, a cat, or a wild animal",
            "Mancare": "delicious food, a meal, drinks, or a restaurant setting"
        }

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate()
        
        # 1. PREGATIRE MODEL AI
        if self.model is None:
            # Folosim B-32 pentru viteza pe 8GB RAM
            self.model = SentenceTransformer('clip-ViT-B-32')

        # --- PRE-CALCULARE VECTORI CATEGORII ---
        # Facem asta o singura data la inceputul scanarii pentru a economisi timp
        nume_categorii = list(self.categorii_config.keys())
        prompte_categorii = list(self.categorii_config.values())
        vectori_categorii = self.model.encode(prompte_categorii, normalize_embeddings=True)

        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        try:
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
            existenta = db.cauta_dupa_cale(cale_full)

            # Verificam daca are deja si categoria (existenta[5] conform structurii noi)
            are_cache = existenta and len(existenta) > 6 and existenta[6] and os.path.exists(existenta[6])
            are_vector = existenta and len(existenta) > 4 and existenta[4] is not None
            are_categorie = existenta and len(existenta) > 5 and existenta[5] is not None

            if are_cache and are_vector and are_categorie:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    img_fix = ImageOps.exif_transpose(img)
                    
                    # 2. GENERARE VECTOR AI (CLIP)
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    
                    # --- LOGICA DE AUTO-ORGANIZARE (Zero-Shot) ---
                    # Comparam vectorul pozei cu toate categoriile prin produs scalar (dot product)
                    scoruri = np.dot(vectori_categorii, vector_ai)
                    idx_castigator = np.argmax(scoruri)
                    
                    # Prag de siguranta: daca niciun scor nu e > 0.18, o punem la "Diverse"
                    if scoruri[idx_castigator] > 0.18:
                        categorie_finala = nume_categorii[idx_castigator]
                    else:
                        categorie_finala = "Diverse"

                    # 3. GENERARE CACHE
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
                        'vector_ai': vector_ai,
                        'categorie': categorie_finala # <--- NOUA COLOANA
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
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare procesare AI pentru {nume}: {e}")

            self.progres.emit(current, total)

        self.finalizat.emit()