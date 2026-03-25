import os
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
from sentence_transformers import SentenceTransformer

# Functie pentru conversia coordonatelor GPS in format lizibil
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

    def __init__(self, cale_folder, folder_cache, cale_db, recursiv=False):
        super().__init__()
        self.cale_folder = cale_folder
        self.folder_cache = folder_cache
        self.cale_db = cale_db 
        self.recursiv = recursiv # Retinem daca scanam si subfolderele
        self.running = True
        self.model = None 
        
        # --- CONFIGURATIE CATEGORII (Prompt Ensembling) ---
        self.categorii_config = {
            "Oameni": [
                "a photo of a person", "a portrait of a human", "human faces", 
                "people in a group", "a photograph of a man or a woman"
            ],
            "Natura": [
                "scenery of nature", "forest and trees", "green plants and foliage", 
                "pine tree needles and branches", "landscape photography", 
                "mountains and lakes", "wild vegetation"
            ],
            "Tehnologie": [
                "computer hardware parts", "internal pc components", 
                "hard disk drive and ram sticks", "isolated hardware on white background",
                "electronics with circuit boards", "macro of microchips"
            ],
            "Documente": [
                "a page of a book", "scanned document with paragraphs", 
                "official paper with text", "a full page of writing", 
                "digital file screenshot", "printed document"
            ],
            "Arhitectura": [
                "city buildings and skyscrapers", "residential houses", 
                "urban architecture", "indoor or outdoor structures", 
                "construction and monuments"
            ],
            "Vehicule": [
                "transportation vehicles", "cars and automobiles on the road", 
                "motorcycles", "airplanes in the sky", "trains, boats or ships"
            ],
            "Animale": [
                "a photo of an animal", "wildlife and pets", 
                "mammals, birds or reptiles", "living creatures", 
                "animal fur, feathers or scales"
            ],
            "Mancare": [
                "a dish of food", "delicious meal on a table", 
                "cooking and gourmet photography", "fruit and vegetables", 
                "snacks and beverages"
            ],
            "Evenimente": [
                "social event or party", "wedding and celebration", 
                "christmas tree with ornaments and decorations", 
                "concert or festival atmosphere", "holiday festivities"
            ]
        }

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate(self.cale_db) 
        
        if self.model is None:
            self.model = SentenceTransformer('clip-ViT-B-32')

        # Pregatire vectori categorii pentru clasificare rapida
        nume_categorii = list(self.categorii_config.keys())
        vectori_reprezentativi = []
        for nume in nume_categorii:
            v_prompte = self.model.encode(self.categorii_config[nume], normalize_embeddings=True)
            v_mediu = np.mean(v_prompte, axis=0)
            v_mediu = v_mediu / np.linalg.norm(v_mediu)
            vectori_reprezentativi.append(v_mediu)
        vectori_categorii = np.array(vectori_reprezentativi)

        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        fisiere_totale = []
        
        # Filtru pentru foldere de sistem ca sa nu blocam PC-ul
        foldere_ignore = ["Windows", "Program Files", "AppData", ".cache"]

        try:
            if self.recursiv:
                # --- CAZUL RECURSIV: Intra in toate subfolderele ---
                for radacina, directoare, fisiere_nume in os.walk(self.cale_folder):
                    # Spunem os.walk sa ignore folderele de sistem
                    directoare[:] = [d for d in directoare if d not in foldere_ignore]
                    
                    for nume in fisiere_nume:
                        if nume.lower().endswith(formate):
                            cale_completa = os.path.join(radacina, nume).replace('\\', '/')
                            fisiere_totale.append(cale_completa)
            else:
                # --- CAZUL SIMPLE: Doar pozele din folderul principal ---
                for nume in os.listdir(self.cale_folder):
                    cale_f = os.path.join(self.cale_folder, nume).replace('\\', '/')
                    # Verificam daca e fisier (nu folder) si daca are extensia corecta
                    if os.path.isfile(cale_f) and nume.lower().endswith(formate):
                        fisiere_totale.append(cale_f)
            
            fisiere_totale.sort()
            
        except Exception as e:
            print(f"Eroare la cautarea fisierelor: {e}")
            return

        total = len(fisiere_totale)
        if not os.path.exists(self.folder_cache): 
            os.makedirs(self.folder_cache)

        # --- LOOP PROCESARE IMAGINI ---
        for i, cale_full in enumerate(fisiere_totale):
            if not self.running: break
            current = i + 1
            nume_fisier = os.path.basename(cale_full)
            
            # Verificam daca imaginea este deja procesata corect in baza de date
            existenta = db.cauta_dupa_cale(cale_full)
            if existenta and existenta[10] and os.path.exists(existenta[10]) and existenta[12] is not None:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    # Ignoram imaginile prea mici
                    if img.width < 200 or img.height < 200:
                        self.progres.emit(current, total)
                        continue
                        
                    # Corectie orientare si conversie la RGB
                    img_fix = ImageOps.exif_transpose(img)
                    if img_fix.mode != "RGB": img_fix = img_fix.convert("RGB")
                    
                    # Analiza AI (Embeddings + Clasificare)
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    scoruri = np.dot(vectori_categorii, vector_ai)
                    idx = np.argmax(scoruri)
                    
                    # Pragul reglat la 0.20 pentru a fi putin mai permisiv
                    categorie_finala = nume_categorii[idx] if scoruri[idx] > 0.20 else "Diverse"

                    # Creare si salvare Thumbnail
                    nume_cache = f"cache_{current}_{nume_fisier}.png"
                    cale_cache_finala = os.path.join(self.folder_cache, nume_cache).replace('\\', '/')
                    
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((256, 256)) 
                    img_thumb.save(cale_cache_finala, "PNG")

                    date_info = {
                        'cale': cale_full, 'nume': nume_fisier, 'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache_finala, 'vector_ai': vector_ai, 'categorie': categorie_finala
                    }
                    
                    # Extragere Metadate EXIF
                    exif = img._getexif()
                    if exif:
                        for id_tag, val in exif.items():
                            n_tag = TAGS.get(id_tag, id_tag)
                            if n_tag == "Make": date_info['marca'] = val
                            if n_tag == "Model": date_info['model'] = val
                            if n_tag == "DateTimeOriginal": date_info['data'] = val
                            if n_tag == "GPSInfo":
                                info_g = {}
                                for t in val:
                                    s_tag = GPSTAGS.get(t, t)
                                    info_g[s_tag] = val[t]
                                if "GPSLatitude" in info_g:
                                    lat = converteste_gps(info_g["GPSLatitude"])
                                    lon = converteste_gps(info_g["GPSLongitude"])
                                    date_info['gps'] = f"Lat: {lat} | Lon: {lon}"

                    # Salvare definitiva in SQLite
                    db.salveaza_sau_actualizeaza(date_info)
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare procesare {cale_full}: {e}")

            self.progres.emit(current, total)
            
        self.finalizat.emit()