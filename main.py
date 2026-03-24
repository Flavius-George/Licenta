import sys
import os
import numpy as np
import faiss
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon
from PySide6.QtCore import Qt, QSize, QDir, QStandardPaths, QSortFilterProxyModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileSystemModel

from database import ManagerBazaDate
from scanner_worker import ScannerWorker
from worker import ProcesorImagine
from sentence_transformers import SentenceTransformer

# 1. INITIALIZARE COORDONATORI
db_manager = ManagerBazaDate()
# Modelul pentru cautare (il incarcam global pentru a fi gata de search)
print("Se incarca creierul AI pentru cautare...")
model_ai = SentenceTransformer('clip-ViT-B-32')

# FAISS Index Local pentru cautare instanta
dimensiune_vector = 512
index_faiss = faiss.IndexFlatIP(dimensiune_vector)
mapare_cai = [] # Retinem ordinea: pozitia in FAISS -> calea fisierului

def incarca_index_faiss():
    """Citeste amprentele AI din baza de date si populeaza motorul de cautare."""
    global index_faiss, mapare_cai
    index_faiss.reset()
    mapare_cai = []
    
    date_vectori = db_manager.obtine_toti_vectorii()
    if not date_vectori:
        return

    vectori_lista = []
    for cale, v_numpy in date_vectori:
        vectori_lista.append(v_numpy)
        mapare_cai.append(cale)
    
    if vectori_lista:
        v_final = np.array(vectori_lista).astype('float32')
        index_faiss.add(v_final)
    print(f"Index FAISS pregatit cu {index_faiss.ntotal} imagini.")

def actualizeaza_smart_albums():
    """Citeste din DB cate poze sunt in fiecare categorie si updateaza lista din stanga."""
    # Gasim widget-ul de tip lista din interfata
    smart_list = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")
    if not smart_list: 
        return

    # Lista de categorii trebuie sa fie identica cu cheile din ScannerWorker
    categorii = [
        "Oameni", 
        "Natura", 
        "Tehnologie", 
        "Documente", 
        "Arhitectura", 
        "Vehicule", 
        "Animale", 
        "Mancare", 
        "Evenimente", 
        "Diverse"
    ]
    
    # Stergem ce era inainte in lista vizuala ca sa nu se dubleze la fiecare refresh
    smart_list.clear() 

    for cat in categorii:
        # Apelam functia din database.py care face SELECT COUNT(*)
        total = db_manager.numara_per_categorie(cat)
        
        # Cream textul de afisat: "Natura (12)"
        item_text = f"{cat} ({total})"
        
        # Cream un element nou pentru lista
        item = QtWidgets.QListWidgetItem(item_text)
        
        # Optional: Poti seta un font bold daca total > 0 ca sa iasa in evidenta
        if total > 0:
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            # Poti pune si o iconita standard de sistem daca vrei
            # item.setIcon(window.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))

        # Adaugam elementul efectiv in widget-ul de pe interfata
        smart_list.addItem(item)
        
    print(f"[UI] Numaratoare categorii actualizata. S-au procesat {len(categorii)} etichete.")


def cand_apas_pe_smart_album(item):
    global vizualizare_activa
    vizualizare_activa="smart"
    """Afiseaza in galerie toate pozele din categoria selectata, din orice folder."""
    text_complet = item.text() 
    categorie = text_complet.split(" (")[0] # Luam "Documente" din "Documente (1)"
    
    # 1. Resetam filtrul de cautare ca sa nu ascunda noile poze
    proxy_model.setFilterFixedString("")
    
    # 2. Curatam galeria curenta
    model_galerie.clear()
    
    # 3. Luam de la baza de date TOATE caile care apartin acestei categorii
    cai_fisiere = db_manager.obtine_cai_dupa_categorie(categorie)
    
    if not cai_fisiere:
        print(f"Nu am gasit poze pentru categoria: {categorie}")
        return

    # 4. Adaugam fiecare poza gasita in galerie (exact ca la click pe folder)
    for cale_full in cai_fisiere:
        if not os.path.exists(cale_full):
            continue
            
        nume_fisier = os.path.basename(cale_full)
        item_galerie = QStandardItem(nume_fisier)
        
        # Cautam daca avem cache (iconita) in DB pentru aceasta poza
        date_db = db_manager.cauta_dupa_cale(cale_full)
        
        cale_iconita = cale_full # Default e poza originala
        if date_db and len(date_db) > 10 and date_db[10]: # Coloana cale_cache
            if os.path.exists(date_db[10]):
                cale_iconita = date_db[10]

        item_galerie.setData(QIcon(cale_iconita), Qt.ItemDataRole.DecorationRole)
        item_galerie.setData(cale_full, Qt.ItemDataRole.UserRole)
        model_galerie.appendRow(item_galerie)
        
    print(f"[Smart Album] Am incarcat {len(cai_fisiere)} imagini pentru {categorie}")

# --- FUNCTII INTERFATA ---

def actualizeaza_panou_dreapta(date, pixmap):
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    info_label = window.findChild(QtWidgets.QLabel, "infoLabel")
    if not preview_label or not info_label: return

    if pixmap and not pixmap.isNull():
        pixmap_scalat = pixmap.scaled(preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        preview_label.setPixmap(pixmap_scalat)

    detalii = [
        f"<b>Fisier:</b> {date.get('nume', '---')}",
        f"<b>Marime:</b> {date.get('mb', 0)} MB",
        f"<b>Rezolutie:</b> {date.get('rezolutie', '---')}"
    ]
    if date.get('marca'): detalii.append(f"<b>Aparat:</b> {date['marca']} {date.get('model','')}")
    if date.get('gps'): detalii.append(f"<b>Locatie:</b> {date['gps']}")
    
    info_label.setText("<br>".join(detalii))
    info_label.setWordWrap(True)

procesor_activ = None
scanner_activ = None
index_folder_curent = None
vizualizare_activa = "folder"

def cand_selectez_o_imagine(index):
    global procesor_activ
    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    if not cale_fisier: return

    date_db = db_manager.cauta_dupa_cale(cale_fisier)
    if date_db:
        info_d = {
            'cale': date_db[1], 'nume': date_db[2], 'format': date_db[3],
            'rezolutie': date_db[4], 'mb': date_db[5], 'marca': date_db[6],
            'model': date_db[7], 'data': date_db[8], 'gps': date_db[9]
        }
        if date_db[10] and os.path.exists(date_db[10]):
            actualizeaza_panou_dreapta(info_d, QtGui.QPixmap(date_db[10]))
            return

    if procesor_activ and procesor_activ.isRunning():
        procesor_activ.terminate()
    
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    procesor_activ = ProcesorImagine(cale_fisier, preview_label.size())
    procesor_activ.gata_procesarea.connect(actualizeaza_panou_dreapta)
    procesor_activ.start()

def actualizeaza_iconita_live(actual):
    """Update iconita dupa ce AI-ul a terminat de procesat o poza."""
    index_model = model_galerie.index(actual - 1, 0)
    if not index_model.isValid(): return
    
    cale_originala = index_model.data(Qt.ItemDataRole.UserRole)
    date = db_manager.cauta_dupa_cale(cale_originala)
    
    if date and len(date) > 10 and date[10]:
        if os.path.exists(date[10]):
            model_galerie.setData(index_model, QIcon(date[10]), Qt.ItemDataRole.DecorationRole)

def updateaza_status_progres(curent, total):
    """Afiseaza progresul scanarii in bara de jos a ferestrei."""
    if window.statusBar():
        window.statusBar().showMessage(f"AI Analiza: {curent}/{total}")

def cand_apas_pe_folder(index):
    global scanner_activ, index_folder_curent, vizualizare_activa
    vizualizare_activa = "folder"
    index_folder_curent = index
    cale_folder = tree_model.filePath(index)
    
    if not os.path.isdir(cale_folder): 
        cale_folder = os.path.dirname(cale_folder)
    
    model_galerie.clear()
    formate = ('.png', '.jpg', '.jpeg', '.bmp')
    fisiere_de_afisat = []

    # --- LOGICA DE SELECTIE RECURSIVA ---
    check_recursive = window.findChild(QtWidgets.QCheckBox, "checkRecursive")
    
    if check_recursive and check_recursive.isChecked():
        # Scanam recursiv toate subfolderele (Flat View)
        for radacina, directoare, numele_fisiere in os.walk(cale_folder):
            for nume in numele_fisiere:
                if nume.lower().endswith(formate):
                    fisiere_de_afisat.append(os.path.join(radacina, nume))
    else:
        # Scanam doar folderul curent
        try:
            nume_fisiere = os.listdir(cale_folder)
            for nume in nume_fisiere:
                if nume.lower().endswith(formate):
                    fisiere_de_afisat.append(os.path.join(cale_folder, nume))
        except Exception as e: 
            print(f"Eroare listare: {e}")

    fisiere_de_afisat.sort()

    # Construim galeria vizuala (fara nicio simplificare)
    for cale_full in fisiere_de_afisat:
        nume_fisier = os.path.basename(cale_full)
        item = QStandardItem(nume_fisier)
        date_ex = db_manager.cauta_dupa_cale(cale_full)
        
        cale_icon = cale_full
        if date_ex and len(date_ex) > 10 and date_ex[10] and os.path.exists(date_ex[10]):
            cale_icon = date_ex[10]

        item.setData(QIcon(cale_icon), Qt.ItemDataRole.DecorationRole)
        item.setData(cale_full, Qt.ItemDataRole.UserRole)
        model_galerie.appendRow(item)

    # --- CONFIGURARE SCANNER AI ---
    if scanner_activ and scanner_activ.isRunning():
        scanner_activ.stop()
        scanner_activ.wait()

    scanner_activ = ScannerWorker(cale_folder)
    
    # Conectam semnalele la functii clasice (fara lambda)
    scanner_activ.imagine_reparata.connect(actualizeaza_iconita_live)
    scanner_activ.progres.connect(updateaza_status_progres) # <--- Functie clasica
    
    # Conexiunile de finalizare
    scanner_activ.finalizat.connect(incarca_index_faiss) 
    scanner_activ.finalizat.connect(actualizeaza_smart_albums) # <--- Refresh la cifrele din sidebar

    scanner_activ.start()


# --- LOGICA DE CAUTARE AI ---

def execut_cautare_ai():
    """Functia magica: cauta poze semantic folosind un prompt optimizat."""
    text_original = search_bar.text().strip()
    
    # 1. Resetare: Daca bara e goala, aratam tot
    if not text_original:
        proxy_model.setFilterFixedString("")
        print("[AI Search] Resetare filtru.")
        return

    # --- OPTIMIZARE PROMPT ---
    # CLIP a fost antrenat pe descrieri, nu pe cuvinte cheie. 
    # Adaugarea "a photo of" ajuta modelul sa identifice mult mai bine subiectul.
    text_optimizat = f"a photo of {text_original}"
    print(f"\n[AI Search] Cautare originala: '{text_original}'")
    print(f"[AI Search] Prompt optimizat pentru AI: '{text_optimizat}'")

    # 2. Transformam textul optimizat in vector (Embedding)
    vector_text = model_ai.encode([text_optimizat], normalize_embeddings=True).astype('float32')

    # 3. Cautam in FAISS
    k_cerut = min(5, index_faiss.ntotal)
    if k_cerut == 0: 
        print("[AI Search] Indexul FAISS este gol. Scaneaza un folder mai intai!")
        return
    
    distante, indexuri = index_faiss.search(vector_text, k_cerut)
    
    # 4. Filtrare dupa SCOR
    # Cu "a photo of", scorurile cresc, deci 0.21 e un prag foarte sigur (safe).
    prag_relevanta = 0.22 
    nume_gasite = []
    
    print(f"[AI Search] Rezultate gasite (Prag > {prag_relevanta}):")
    
    for i, idx in enumerate(indexuri[0]):
        if idx != -1:
            scor = distante[0][i]
            if scor > prag_relevanta:
                cale_completa = mapare_cai[idx]
                nume_fisier = os.path.basename(cale_completa)
                
                nume_gasite.append(QtCore.QRegularExpression.escape(nume_fisier))
                print(f"   - {nume_fisier} | Scor: {scor:.4f} [ADMIS]")
            else:
                cale_reapinsa = os.path.basename(mapare_cai[idx])
                print(f"   - {cale_reapinsa} | Scor: {scor:.4f} [RESPINS - sub prag]")

    # 5. Aplicam FILTRUL vizual
    if nume_gasite:
        pattern = "^(" + "|".join(nume_gasite) + ")$"
        regex = QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.CaseInsensitiveOption)
        
        proxy_model.setFilterRegularExpression(regex)
        proxy_model.setFilterKeyColumn(0) 
        print(f"[AI Search] Galeria a fost filtrata. ({len(nume_gasite)} rezultate)")
    else:
        proxy_model.setFilterFixedString("___NIMIC_GASIT___")
        print("[AI Search] Niciun rezultat nu a trecut pragul de relevanta.")

def aplic_filtrare_simpla(text):
    """Filtrare instanta dupa nume (cand scrii)."""
    proxy_model.setFilterFixedString(text)
    proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

# --- FUNCTII NOI PENTRU ALBUME SMART ---

def actualizeaza_smart_albums():
    smart_list = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")
    if not smart_list: return

    # Lista trebuie sa aiba aceleasi nume ca cheile din ScannerWorker
    categorii = ["Oameni", "Natura", "Tehnologie", "Documente", "Arhitectura", "Vehicule", "Animale", "Mancare", "Evenimente", "Diverse"]
    
    smart_list.clear() 
    for cat in categorii:
        total = db_manager.numara_per_categorie(cat)
        item_text = f"{cat} ({total})"
        item = QtWidgets.QListWidgetItem(item_text)
        smart_list.addItem(item)

def cand_apas_pe_smart_album(item):
    """Afiseaza in galerie toate pozele din categoria selectata, din orice folder."""
    text_complet = item.text() 
    categorie = text_complet.split(" (")[0] # Luam "Documente" din "Documente (1)"
    
    # 1. Resetam filtrul de cautare ca sa nu ascunda noile poze
    proxy_model.setFilterFixedString("")
    
    # 2. Curatam galeria curenta
    model_galerie.clear()
    
    # 3. Luam de la baza de date TOATE caile care apartin acestei categorii
    cai_fisiere = db_manager.obtine_cai_dupa_categorie(categorie)
    
    if not cai_fisiere:
        print(f"Nu am gasit poze pentru categoria: {categorie}")
        return

    # 4. Adaugam fiecare poza gasita in galerie (exact ca la click pe folder)
    for cale_full in cai_fisiere:
        if not os.path.exists(cale_full):
            continue
            
        nume_fisier = os.path.basename(cale_full)
        item_galerie = QStandardItem(nume_fisier)
        
        # Cautam daca avem cache (iconita) in DB pentru aceasta poza
        date_db = db_manager.cauta_dupa_cale(cale_full)
        
        cale_iconita = cale_full # Default e poza originala
        if date_db and len(date_db) > 10 and date_db[10]: # Coloana cale_cache
            if os.path.exists(date_db[10]):
                cale_iconita = date_db[10]

        item_galerie.setData(QIcon(cale_iconita), Qt.ItemDataRole.DecorationRole)
        item_galerie.setData(cale_full, Qt.ItemDataRole.UserRole)
        model_galerie.appendRow(item_galerie)
        
    print(f"[Smart Album] Am incarcat {len(cai_fisiere)} imagini pentru {categorie}")

# --- LANSAREA ---
app = QtWidgets.QApplication(sys.argv)
loader = QUiLoader()
window = loader.load("interfata.ui", None)

# Incarcam datele AI existente in FAISS la pornire
incarca_index_faiss()

view_galerie = window.findChild(QtWidgets.QListView, "photoView")
model_galerie = QStandardItemModel()
proxy_model = QSortFilterProxyModel()
proxy_model.setSourceModel(model_galerie)

view_galerie.setModel(proxy_model)
view_galerie.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
# --- LINIILE DE REPARARE ---
# Aceasta linie face pozele sa se rearanjeze automat la resize/fullscreen:
view_galerie.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)

# Previne suprapunerea pozelor si le tine intr-o grila ordonata:
view_galerie.setMovement(QtWidgets.QListView.Movement.Static)
view_galerie.setSpacing(10) # Adauga putin spatiu intre poze sa nu fie lipite
# ---------------------------
view_galerie.setIconSize(QSize(130, 130))
view_galerie.setGridSize(QSize(160, 180))
view_galerie.clicked.connect(cand_selectez_o_imagine)

tree_view = window.findChild(QtWidgets.QTreeView, "treeViewFolders")
tree_model = QFileSystemModel()
tree_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
tree_model.setNameFilters(["*.png", "*.jpg", "*.jpeg", "*.bmp"])
tree_model.setNameFilterDisables(False)
cale_start = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
tree_model.setRootPath(cale_start)
tree_view.setModel(tree_model)
tree_view.setRootIndex(tree_model.index(cale_start))
for i in range(1, 4): tree_view.hideColumn(i)
tree_view.clicked.connect(cand_apas_pe_folder)
#-----------------------------------------------------------------------------------
def refresh_galerie_la_bifa():
    """Cauta din nou pozele daca am schimbat setarea de recursivitate."""
    global index_folder_curent, vizualizare_activa
    # Daca am apasat deja pe un folder inainte, re-apelam functia cu acelasi index
    if vizualizare_activa == "folder" and index_folder_curent:
        cand_apas_pe_folder(index_folder_curent)

# Gasim checkbox-ul si il conectam
check_recursive = window.findChild(QtWidgets.QCheckBox, "checkRecursive")
if check_recursive:
    # stateChanged se declanseaza cand bifezi sau debifezi
    check_recursive.stateChanged.connect(refresh_galerie_la_bifa)

search_bar = window.findChild(QtWidgets.QLineEdit, "searchBar")
if search_bar:
    # Cand doar scrii, cauta dupa nume (rapid)
    search_bar.textChanged.connect(aplic_filtrare_simpla)
    # Cand apesi ENTER, porneste AI-ul (semantic)
    search_bar.returnPressed.connect(execut_cautare_ai)
#----------------------------------------------------------------------------------

# Gasim lista de albume in UI
smart_album_view = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")

if smart_album_view:
    # Conectam click-ul la functia de filtrare
    smart_album_view.itemClicked.connect(cand_apas_pe_smart_album)

# Prima numaratoare la deschiderea aplicatiei
actualizeaza_smart_albums()

window.show()
sys.exit(app.exec())