import io
import os
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS, GPSTAGS


def converti_gps(valoare) -> str:
    """Converteste coordonate EXIF (fractii) in string grad/min/sec."""
    try:
        grade   = float(valoare[0])
        minute  = float(valoare[1])
        secunde = float(valoare[2])
        return f"{grade:.2f}°, {minute:.2f}', {secunde:.2f}\""
    except Exception:
        return str(valoare)


class ProcesorImagine(QThread):
    """
    Worker thread pentru procesarea live a unei imagini selectate.

    Folosit ca fallback atunci cand o imagine nu se afla inca in baza de date.
    Emite semnalul `gata_procesarea` cu datele EXIF si preview-ul scalat.
    """

    gata_procesarea = Signal(dict, QPixmap)

    def __init__(self, cale_poza: str, dimensiune_preview):
        super().__init__()
        self.cale_poza = cale_poza
        self.dimensiune_preview = dimensiune_preview

    def run(self):
        date_info: dict = {}
        pixmap_final = QPixmap()

        try:
            with Image.open(self.cale_poza) as img:
                # 1. Corectam rotatia din EXIF
                img_fix = ImageOps.exif_transpose(img)

                # 2. Convertim pentru Qt
                if img_fix.mode in ("P", "RGBA"):
                    img_fix = img_fix.convert("RGB")

                buf = io.BytesIO()
                img_fix.save(buf, format="PNG")
                pixmap_final.loadFromData(buf.getvalue())
                pixmap_final = pixmap_final.scaled(
                    self.dimensiune_preview,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

                # 3. Metadate de baza
                date_info["format"]     = img.format
                date_info["rezolutie"]  = f"{img_fix.width} x {img_fix.height}"

                # 4. Metadate EXIF
                exif_raw = img._getexif()
                if exif_raw:
                    for id_tag, val in exif_raw.items():
                        tag = TAGS.get(id_tag, id_tag)
                        if tag == "Make":
                            date_info["marca"] = str(val).strip()
                        elif tag == "Model":
                            date_info["model"] = str(val).strip()
                        elif tag == "DateTimeOriginal":
                            date_info["data"] = str(val)
                        elif tag == "GPSInfo":
                            info_gps = {GPSTAGS.get(t, t): val[t] for t in val}
                            if "GPSLatitude" in info_gps and "GPSLongitude" in info_gps:
                                lat = converti_gps(info_gps["GPSLatitude"])
                                lon = converti_gps(info_gps["GPSLongitude"])
                                date_info["gps"] = f"Lat: {lat} | Lon: {lon}"
                            else:
                                date_info["gps"] = "Disponibil (fara coordonate)"

            date_info["mb"]   = round(os.path.getsize(self.cale_poza) / (1024 * 1024), 2)
            date_info["nume"] = os.path.basename(self.cale_poza)
            date_info["cale"] = self.cale_poza

        except Exception as e:
            print(f"[ProcesorImagine] Eroare la '{self.cale_poza}': {e}")

        self.gata_procesarea.emit(date_info, pixmap_final)
