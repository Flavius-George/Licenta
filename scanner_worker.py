import os
import numpy as np
import pickle
from PySide6.QtCore import QThread, Signal
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
        self.recursiv = recursiv
        self.running = True
        self.model = None 
        
        # --- CONFIGURATIE CATEGORII (Sincronizat cu TreeWidget) ---
        self.lista_categorii = [
            "Oameni", "Mancare", "Animale", 
            "Nunti", "Petreceri", "Sarbatori",
            "Documente Clasice", "Diagrame & Scheme", "Baze de Date", 
            "Hardware", "Interfete Software", "Screenshots Cod",
            "Natura", "Arhitectura", "Vehicule", "Diverse"
        ]

        # Prompt-uri optimizate pentru a ajuta AI-ul sa identifice corect
        self.prompts_categorii = {
            "Oameni": "a photo of people, portraits, or faces",
            "Mancare": "a photo of food, meals, or appetizers",
            "Animale": "a photo of a pet, wild animal, or domestic creature",
            "Nunti": "a photo of a wedding, bride, groom, or wedding ceremony",
            "Petreceri": "a photo of a party, social gathering, or celebration",
            "Sarbatori": "a photo of Christmas, Easter, or holiday decorations",
            "Documente Clasice": "a photo of a printed document, a page with text, a scan of a book or a paper invoice",
            "Diagrame & Scheme": "a technical diagram, flowchart, petri net, or logical scheme with nodes and arrows",
            "Baze de Date": "a database schema, ER diagram, or a table structure with boxes and relations",
            "Hardware": "a photo of computer hardware, motherboards, or electronic components",
            "Interfete Software": "a screenshot of a mobile app or a computer software user interface",
            "Screenshots Cod": "a screenshot of programming code with syntax highlighting in a text editor",
            "Natura": "a photo of a natural landscape, forest, mountains, or sea",
            "Arhitectura": "a photo of a building, urban skyline, or city street",
            "Vehicule": "a photo of a car, truck, motorcycle, or airplane",
            "Diverse": "a random object, abstract image, or unclassified content"
        }

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate(self.cale_db) 
        
        if self.model is None:
            print("[Scanner] Se incarca modelul CLIP...")
            self.model = SentenceTransformer('clip-ViT-B-32')

        # --- PREGATIRE VECTORI CATEGORII ---
        # Generam "amprenta" matematica pentru fiecare categorie
        vectori_reprezentativi = []
        for nume in self.lista_categorii:
            prompt = self.prompts_categorii[nume]
            # Generam vectorul pentru textul descriptiv
            v_text = self.model.encode([prompt], normalize_embeddings=True)[0]
            vectori_reprezentativi.append(v_text)
        
        vectori_categorii = np.array(vectori_reprezentativi)

        # --- CAUTARE FISIERE ---
        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        fisiere_totale = []
        foldere_ignore = ["Windows", "Program Files", "AppData", ".cache"]

        try:
            if self.recursiv:
                for radacina, directoare, fisiere_nume in os.walk(self.cale_folder):
                    directoare[:] = [d for d in directoare if d not in foldere_ignore]
                    for nume in fisiere_nume:
                        if nume.lower().endswith(formate):
                            cale_completa = os.path.join(radacina, nume).replace('\\', '/')
                            fisiere_totale.append(cale_completa)
            else:
                for nume in os.listdir(self.cale_folder):
                    cale_f = os.path.join(self.cale_folder, nume).replace('\\', '/')
                    if os.path.isfile(cale_f) and nume.lower().endswith(formate):
                        fisiere_totale.append(cale_f)
            
            fisiere_totale.sort()
        except Exception as e:
            print(f"Eroare cautare: {e}")
            return

        total = len(fisiere_totale)
        if total == 0:
            self.finalizat.emit()
            return

        # --- LOOP PROCESARE ---
        for i, cale_full in enumerate(fisiere_totale):
            if not self.running: break
            current = i + 1
            nume_fisier = os.path.basename(cale_full)
            
            # Verificam daca e deja procesata
            existenta = db.cauta_dupa_cale(cale_full)
            if existenta and existenta[10] and os.path.exists(existenta[10]) and existenta[12] is not None:
                self.progres.emit(current, total)
                continue

            try:
                with Image.open(cale_full) as img:
                    if img.width < 100 or img.height < 100: 
                        continue
                        
                    img_fix = ImageOps.exif_transpose(img)
                    if img_fix.mode != "RGB": img_fix = img_fix.convert("RGB")
                    
                    # 1. Genereaza Vector Imagine (Embedding)
                    vector_ai = self.model.encode(img_fix, normalize_embeddings=True)
                    
                    # 2. Clasificare (Similitudine intre vectorul pozei si vectorii categoriilor)
                    scoruri = np.dot(vectori_categorii, vector_ai)
                    idx_max = np.argmax(scoruri)
                    
                    # Prag de siguranta: daca scorul e prea mic, punem la Diverse
                    if scoruri[idx_max] > 0.18:
                        categorie_finala = self.lista_categorii[idx_max]
                    else:
                        categorie_finala = "Diverse"

                    # 3. Creare Thumbnail
                    nume_cache = f"cache_{i}_{nume_fisier}.png"
                    cale_cache_finala = os.path.join(self.folder_cache, nume_cache).replace('\\', '/')
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((256, 256)) 
                    img_thumb.save(cale_cache_finala, "PNG")

                    # 4. Colectare Metadate
                    date_info = {
                        'cale': cale_full, 'nume': nume_fisier, 'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache_finala, 'vector_ai': vector_ai, 
                        'categorie': categorie_finala,
                        'marca': 'Necunoscut', 'model': 'Necunoscut', 'data': '---', 'gps': ''
                    }
                    
                    exif = img._getexif()
                    if exif:
                        for id_tag, val in exif.items():
                            n_tag = TAGS.get(id_tag, id_tag)
                            if n_tag == "Make": date_info['marca'] = str(val).strip()
                            if n_tag == "Model": date_info['model'] = str(val).strip()
                            if n_tag == "DateTimeOriginal": date_info['data'] = str(val)
                            if n_tag == "GPSInfo":
                                info_g = {GPSTAGS.get(t, t): val[t] for t in val}
                                if "GPSLatitude" in info_g:
                                    lat = converteste_gps(info_g["GPSLatitude"])
                                    lon = converteste_gps(info_g["GPSLongitude"])
                                    date_info['gps'] = f"Lat: {lat} | Lon: {lon}"

                    # 5. Salvare in SQLite
                    db.salveaza_sau_actualizeaza(date_info)
                    self.imagine_reparata.emit(current)
                    
            except Exception as e:
                print(f"Eroare la {nume_fisier}: {e}")

            self.progres.emit(current, total)
            
        self.finalizat.emit()