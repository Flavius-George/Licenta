# geocodare.py
import reverse_geocoder as rg
from functools import lru_cache


TRADUCERI_TARI = {
    "RO": "Romania",
    "MD": "Moldova",
    "US": "SUA",
    "GB": "Marea Britanie",
    "DE": "Germania",
    "FR": "Franta",
    "IT": "Italia",
    "ES": "Spania",
    "GR": "Grecia",
    "TR": "Turcia",
    "HU": "Ungaria",
    "BG": "Bulgaria",
    "RS": "Serbia",
    "UA": "Ucraina",
    # adauga ce tari ai nevoie
}

@lru_cache(maxsize=2048)
def geocodeaza_local(lat: float, lon: float) -> dict:
    """
    Transforma coordonate decimale in tara + oras.
    100% offline, ~1ms per apel, cu cache in memorie.
    
    Returneaza: {"oras": "Galati", "tara": "Romania", "cod": "RO"}
    """
    try:
        rezultat = rg.search((lat, lon), verbose=False)[0]
        cod_tara = rezultat.get("cc", "??")
        return {
            "oras":    rezultat.get("name", "Necunoscut"),
            "tara":    TRADUCERI_TARI.get(cod_tara, rezultat.get("cc", "Necunoscut")),
            "cod":     cod_tara,
        }
    except Exception:
        return {"oras": "Necunoscut", "tara": "Necunoscut", "cod": "??"}


def gps_exif_la_decimal(valoare_exif, ref: str) -> float | None:
    """
    Converteste coordonata EXIF (grade/minute/secunde ca fractii)
    in grade decimale. ref = 'N'/'S' pentru lat, 'E'/'W' pentru lon.
    """
    try:
        grade   = float(valoare_exif[0])
        minute  = float(valoare_exif[1])
        secunde = float(valoare_exif[2])
        decimal = grade + minute / 60 + secunde / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None