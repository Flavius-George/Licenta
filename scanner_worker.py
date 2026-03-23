import os
from PySide6.QtCore import QThread, Signal, Qt
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS

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
    progres = Signal(int, int)
    finalizat = Signal()

    def __init__(self, cale_folder):
        super().__init__()
        self.cale_folder = cale_folder
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        from database import ManagerBazaDate
        db = ManagerBazaDate()
        
        formate = ('.png', '.jpg', '.jpeg', '.bmp')
        try:
            # --- MODIFICARE CRUCIALA: SORTARE ALFABETICA ---
            # Sortam lista pentru a fi identica cu cea din model_galerie (UI)
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
            
            cale_full = os.path.join(self.cale_folder, nume)
            existenta = db.cauta_dupa_cale(cale_full)

            # Verificam daca avem deja cache valid
            if existenta and len(existenta) > 10 and existenta[10]:
                if os.path.exists(existenta[10]):
                    # Trimitem progresul chiar daca dam skip, pentru a tine indexul sincronizat
                    self.progres.emit(i + 1, total)
                    continue

            try:
                with Image.open(cale_full) as img:
                    # 1. Rotire corecta
                    img_fix = ImageOps.exif_transpose(img)
                    
                    nume_cache = f"cache_{nume}.png"
                    cale_cache = os.path.join(folder_cache, nume_cache)
                    
                    # 2. Generare thumbnail din imaginea deja rotita
                    img_thumb = img_fix.copy()
                    img_thumb.thumbnail((1024, 1024)) 
                    img_thumb.save(cale_cache, "PNG")

                    date_info = {
                        'cale': cale_full,
                        'nume': nume,
                        'format': img.format,
                        'rezolutie': f"{img_fix.width}x{img_fix.height}",
                        'mb': round(os.path.getsize(cale_full) / (1024*1024), 2),
                        'cale_cache': cale_cache
                    }
                    
                    # 3. Extragere EXIF pentru baza de date
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
            except Exception as e:
                print(f"Eroare scanare {nume}: {e}")

            # Trimitem semnalul ca am terminat o poza (i + 1 devine 'actual' in UI)
            self.progres.emit(i + 1, total)

        self.finalizat.emit()