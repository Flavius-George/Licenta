import os
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer

# Funcție pentru conversia coordonatelor GPS
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
        
        # --- CONFIGURAȚIE PROMPTE (Prompt Ensembling în Engleză) ---
        self.categorii_config = {
            "Oameni": [
                "a photo of a person", 
                "a portrait of a man or a woman", 
                "a group of people standing together",
                "human faces and people"
            ],
            "Natura": [
                "a natural landscape with no buildings", 
                "mountains, forests, trees or lakes", 
                "a beautiful view of nature",
                "outdoor scenery with plants and sky"
            ],
            "Tehnologie": [
                "computer hardware and electronic components", 
                "printed circuit boards and microchips", 
                "electronic gadgets and smartphones",
                "modern technology and devices"
            ],
            "Documente": [
                "a screenshot of digital text", 
                "a scanned paper or a document", 
                "black text on a white background",
                "a page from a book or an article"
            ],
            "Arhitectura": [
                "city buildings and architecture", 
                "houses and residential buildings", 
                "urban street view and skyscrapers",
                "interior or exterior of a building"
            ],
            "Vehicule": [
                "cars, trucks or motorcycles on the road", 
                "airplanes, trains or boats", 
                "transportation vehicles",
                "a photo of a vehicle"
            ],
            "Animale": [
                "a photo of a pet like a dog or a cat", 
                "wildlife animals in their habitat", 
                "birds, insects or mammals",
                "a close up of an animal"
            ],
            "Mancare": [
                "a close-up of food on a plate", 
                "a delicious meal or snacks", 
                "culinary photography and cooking",
                "food ingredients and gourmet dishes"
            ],
            "Evenimente": [
                "people celebrating a party or a wedding", 
                "a decorated Christmas tree with lights", 
                "a social gathering or a holiday event",
                "a festive ceremony or a public event"
            ]
        }

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate()
        
        # 1. PREGĂTIRE MODEL AI
        if self.model is None:
            self.model = SentenceTransformer('clip-ViT-B-32')

        # --- PRE-CALCULARE VECTORI MEDII (Prompt Ensembling) ---
        nume_categorii = list(self.categorii_config.keys())
        vectori_reprezentativi = []

        for nume in nume_categorii:
            prompte_lista = self.categorii_config[nume]
            # Generăm vectori pentru toate descrierile din listă
            v_prompte = self.model.encode(prompte_lista, normalize_embeddings=True)
            # Calculăm media lor pentru un "concept" mai stabil
            v_mediu = np.mean(v_prompte, axis=0)
            # Re-normalizăm vectorul mediu (obligatoriu pentru CLIP)
            v_mediu = v_mediu / np.linalg.norm(v_mediu)
            vectori_reprezentativi.append(v_mediu)

        vectori_categorii = np.array(vectori_reprezentativi)

        # 2. SCANARE RECURSIVĂ
        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        fisiere_totale = []
        
        try:
            for radacina, directoare, fisiere_nume in os.walk(self.cale_folder):
                for nume in fisiere_nume:
                    if nume.lower().endswith(formate):
                        cale_completa = os.path.join(radacina, nume)
                        fisiere_totale.append(cale_completa)
            
            fisiere_totale.sort()
        except Exception as e:
            print(f"Eroare la scanarea folderelor: {e}")
            return

        total = len(fisiere_totale)
        folder_cache = os.path.join(os.getcwd(), ".cache")
        if not os.path.exists(folder_cache):
            os.makedirs(folder_cache)

        # 3. PROCESARE FIȘIERE
        for i, cale_full in enumerate(fisiere_totale):
            if not self.running:
                break
            
            current = i + 1
            nume_fisier = os.path.basename(cale_full)
            existenta = db.cauta_dupa_cale(cale_full)

            are_cache = existenta and len(existenta) > 10 and existenta[10] and os.path.exists(existenta[10])
            are_vector = existenta and len(existenta) > 12 and existenta[12] is not None
            are_categorie = existenta and len(existenta) > 11 and existenta[11] is not None

            if are_cache and are_vector and are_categorie:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    img_fix = ImageOps.exif_transpose(img)
                    
                    # Generare vector pentru imagine
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    
                    # Clasificare (Produs Scalar cu vectorii medii)
                    scoruri = np.dot(vectori_categorii, vector_ai)
                    idx_castigator = np.argmax(scoruri)
                    
                    if scoruri[idx_castigator] > 0.18:
                        categorie_finala = nume_categorii[idx_castigator]
                    else:
                        categorie_finala = "Diverse"

                    # Thumbnail
                    nume_cache = f"cache_{current}_{nume_fisier}.png"
                    cale_cache = os.path.join(folder_cache, nume_cache)
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((1024, 1024)) 
                    img_thumb.save(cale_cache, "PNG")

                    date_info = {
                        'cale': cale_full,
                        'nume': nume_fisier,
                        'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache,
                        'vector_ai': vector_ai,
                        'categorie': categorie_finala
                    }
                    
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

                    db.salveaza_sau_actualizeaza(date_info)
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare procesare AI pentru {cale_full}: {e}")

            self.progres.emit(current, total)

        self.finalizat.emit()