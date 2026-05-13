# setup_initial.py
print("Se descarca modelul CLIP (clip-ViT-B-32) ~300MB...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
print("Model salvat local. Aplicatia va functiona offline de acum.")

print("Se descarca baza de date geografica locala...")
import reverse_geocoder as rg
rg.search((45.0, 28.0))  # trigger download dataset ~5MB
print("Gata! Tot sistemul functioneaza 100% offline.")