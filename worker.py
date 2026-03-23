import io
import os
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS # Am adaugat GPSTAGS aici

# --- FUNCTIE HELPER PENTRU GPS ---
def converti_gps(valoare):
    try:
        # EXIF stocheaza datele sub forma de fractii (grade, minute, secunde)
        grade = float(valoare[0])
        minute = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}°, {minute:.2f}', {secunde:.2f}\""
    except:
        return str(valoare)

class ProcesorImagine(QThread):
    gata_procesarea = Signal(dict, QPixmap)

    def __init__(self, cale_poza, dimensiune_preview):
        super().__init__()
        self.cale_poza = cale_poza
        self.dimensiune_preview = dimensiune_preview

    def run(self):
        date_info = {}
        pixmap_final = QPixmap()
        
        try:
            with Image.open(self.cale_poza) as img:
                # 1. REPARAM ROTIREA
                img_corectata = ImageOps.exif_transpose(img)
                
                # 2. CONVERTIM PENTRU QT
                if img_corectata.mode == "P": 
                    img_corectata = img_corectata.convert("RGB")
                
                buffer = io.BytesIO()
                img_corectata.save(buffer, format="PNG")
                
                pixmap_final.loadFromData(buffer.getvalue())
                pixmap_final = pixmap_final.scaled(
                    self.dimensiune_preview, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )

                # 3. EXTRAGEM DATELE EXIF
                date_info['format'] = img.format
                date_info['rezolutie'] = f"{img_corectata.width} x {img_corectata.height}"
                
                exif_raw = img._getexif()
                if exif_raw:
                    for id_tag, valoare in exif_raw.items():
                        nume_tag = TAGS.get(id_tag, id_tag)
                        
                        if nume_tag == "Make": date_info['marca'] = valoare
                        if nume_tag == "Model": date_info['model'] = valoare
                        if nume_tag == "DateTimeOriginal": date_info['data'] = valoare
                        
                        # --- REPARATIA PENTRU GPS AICI ---
                        if nume_tag == "GPSInfo":
                            info_gps = {}
                            # GPSInfo este un dictionar in interiorul altui dictionar
                            for t in valoare:
                                sub_tag = GPSTAGS.get(t, t)
                                info_gps[sub_tag] = valoare[t]
                            
                            if "GPSLatitude" in info_gps and "GPSLongitude" in info_gps:
                                lat = converti_gps(info_gps["GPSLatitude"])
                                lon = converti_gps(info_gps["GPSLongitude"])
                                # Poti adauga si N/S sau E/W daca vrei sa fii super precis
                                date_info['gps'] = f"Lat: {lat} | Lon: {lon}"
                            else:
                                date_info['gps'] = "Disponibil (fara coordonate)"

            stats = os.stat(self.cale_poza)
            date_info['mb'] = round(stats.st_size / (1024 * 1024), 2)
            date_info['nume'] = os.path.basename(self.cale_poza)
            date_info['cale'] = self.cale_poza

        except Exception as e:
            print(f"Eroare in fundal la procesare: {e}")
        
        self.gata_procesarea.emit(date_info, pixmap_final)