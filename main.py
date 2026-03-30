import sys
import os
import pickle
import numpy as np
import faiss
import shutil
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QSize, QDir, QStandardPaths, QSortFilterProxyModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QProgressDialog, QMessageBox, QFileDialog

from database import ManagerBazaDate
from scanner_worker import ScannerWorker
from worker import ProcesorImagine
from sentence_transformers import SentenceTransformer

NumeAplicatie = "GalerieLicentaAI"

# --- CONFIGURARE IERARHIE (Harta pentru UI si Organizare Fizica) ---
# Aceasta structura face legatura intre eticheta AI si folderele de pe disc
STRUCTURA_ALBUME = {
    "A. Viata Personala": {
        "Evenimente": ["Nunti", "Petreceri", "Sarbatori"],
        "Oameni": ["Oameni"],
        "Mancare": ["Mancare"]
    },
    "B. Profesional & Academic": {
        "Documente": ["Documente Clasice", "Diagrame & Scheme", "Baze de Date"],
        "Tehnologie": ["Hardware", "Interfete Software", "Screenshots Cod"]
    },
    "C. Mediu & Obiecte": {
        "Natura": ["Natura", "Animale"],
        "Arhitectura": ["Arhitectura"],
        "Vehicule": ["Vehicule"]
    }
}

# --- INITIALIZARE VARIABILE GLOBALE ---
scanner_activ = None
procesor_activ = None
vizualizare_activa = "librarie"

# --- CONFIGURARE CAI (APPDATA) ---
folder_app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
if not os.path.exists(folder_app_data):
    os.makedirs(folder_app_data)

folder_cache_root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
folder_cache = os.path.join(folder_cache_root, NumeAplicatie, "cache").replace('\\', '/')
if not os.path.exists(folder_cache):
    os.makedirs(folder_cache)

cale_db_finala = os.path.join(folder_app_data, "galerie_licenta.db").replace('\\', '/')
db_manager = ManagerBazaDate(cale_db_finala) 

# --- INITIALIZARE AI ---
print("Se incarca creierul AI pentru cautare...")
model_ai = SentenceTransformer('clip-ViT-B-32')

dimensiune_vector = 512
index_faiss = faiss.IndexFlatIP(dimensiune_vector)
mapare_cai = [] 

# --- FUNCTII LOGICA AI & DATABASE ---

def incarca_index_faiss():
    global index_faiss, mapare_cai
    index_faiss.reset()
    mapare_cai = []
    date_vectori = db_manager.obtine_toti_vectorii()
    if not date_vectori: return
    vectori_lista = []
    for cale, v_numpy in date_vectori:
        v_float = v_numpy.astype('float32')
        # Normalizam pentru similitudine cosinus
        faiss.normalize_L2(v_float.reshape(1, -1))
        vectori_lista.append(v_float)
        mapare_cai.append(cale)
    if vectori_lista:
        v_final = np.array(vectori_lista).astype('float32')
        index_faiss.add(v_final)
        print(f"[AI] Index FAISS pregatit cu {len(mapare_cai)} vectori.")

def actualizeaza_smart_albums():
    """Populeaza QTreeWidget cu ierarhia A/B/C."""
    tree = window.findChild(QtWidgets.QTreeWidget, "smartTreeWidget")
    if not tree: return
    
    tree.clear()
    tree.setHeaderLabel("Organizare Inteligenta")
    tree.setIndentation(20)

    for domeniu, categorii in STRUCTURA_ALBUME.items():
        domeniu_item = QtWidgets.QTreeWidgetItem([domeniu])
        font_d = domeniu_item.font(0); font_d.setBold(True); domeniu_item.setFont(0, font_d)
        tree.addTopLevelItem(domeniu_item)

        for cat, subcategorii in categorii.items():
            cat_item = QtWidgets.QTreeWidgetItem([cat])
            domeniu_item.addChild(cat_item)

            for sub in subcategorii:
                nr = db_manager.numara_per_categorie(sub)
                sub_item = QtWidgets.QTreeWidgetItem([f"{sub} ({nr})"])
                sub_item.setData(0, Qt.UserRole, sub) 
                cat_item.addChild(sub_item)
    tree.expandAll()

# --- FUNCTII LIBRARIE ---

def incarca_sursele_vizual():
    lista_surse_ui = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
    if not lista_surse_ui: return
    lista_surse_ui.clear()
    for s in db_manager.obtine_surse():
        item = QtWidgets.QListWidgetItem(s)
        item.setIcon(window.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        lista_surse_ui.addItem(item)

def adauga_sursa_noua():
    cale = QFileDialog.getExistingDirectory(window, "Selecteaza folderul")
    if cale:
        cale = cale.replace('\\', '/')
        db_manager.adauga_sursa(cale)
        incarca_sursele_vizual()
        check = window.findChild(QtWidgets.QCheckBox, "checkRecursive")
        recursiv = check.isChecked() if check else False
        porneste_scanare_folder(cale, recursiv)

def sterge_sursa_selectata():
    lista = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
    item = lista.currentItem()
    if not item: return
    if QMessageBox.question(window, "Stergere", f"Elimini {item.text()}?") == QMessageBox.Yes:
        db_manager.sterge_sursa_si_imagini(item.text())
        incarca_sursele_vizual()
        afiseaza_toata_libraria()
        actualizeaza_smart_albums()

def afiseaza_toata_libraria():
    global vizualizare_activa
    vizualizare_activa = "librarie"
    populeaza_galeria_cu_cai(db_manager.obtine_toate_caile_existente())

def cand_apas_pe_sursa(item):
    global vizualizare_activa
    vizualizare_activa = "folder"
    toate = db_manager.obtine_toate_caile_existente()
    filtrate = [p for p in toate if p.startswith(item.text())]
    populeaza_galeria_cu_cai(filtrate)

def porneste_scanare_folder(cale, recursiv=False):
    global scanner_activ
    if scanner_activ and scanner_activ.isRunning():
        scanner_activ.stop(); scanner_activ.wait()
    scanner_activ = ScannerWorker(cale, folder_cache, cale_db_finala, recursiv)
    scanner_activ.imagine_reparata.connect(actualizeaza_iconita_live)
    scanner_activ.progres.connect(updateaza_status_progres)
    scanner_activ.finalizat.connect(lambda: (incarca_index_faiss(), actualizeaza_smart_albums(), afiseaza_toata_libraria()))
    scanner_activ.start()

def executa_organizarea_fizica():
    """Creeaza ierarhia de foldere pe disc: An -> Luna -> Locatie -> Categorie AI."""
    # 1. Alegem folderul unde se va face exportul
    destinatie = QFileDialog.getExistingDirectory(window, "Alege unde vrei sa salvezi pozele organizate")
    if not destinatie: return
    
    # 2. Luam datele din baza de date
    date_poze = db_manager.obtine_toate_pentru_organizare()
    if not date_poze: 
        QMessageBox.warning(window, "Atentie", "Baza de date este goala! Scaneaza un folder intai.")
        return

    # 3. Bara de progres pentru utilizator
    progress = QProgressDialog("Se organizeaza colectia pe hard disk...", "Anuleaza", 0, len(date_poze), window)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.show()

    contor_succes = 0
    contor_eroare = 0

    for i, (cale_orig, cat_ai, data_raw, gps_raw) in enumerate(date_poze):
        if progress.wasCanceled(): break
        
        # --- A. LOGICA PENTRU DATA ---
        if data_raw and len(data_raw) >= 7:
            an = data_raw[:4]     # ex: 2025
            luna = data_raw[5:7]   # ex: 03
        else:
            an, luna = "An_Necunoscut", "Luna_Necunoscut"

        # --- B. LOGICA PENTRU LOCATIE (Curatare pentru Windows) ---
        locatie_f = "Fara_Locatie"
        if gps_raw and gps_raw != "Fara GPS":
            # Windows nu suporta caracterele : | / \ * ? etc. in nume de folder
            nume_locatie = gps_raw
            for simbol in [":", "|", "/", "\\", "<", ">", "*", "?", '"']:
                nume_locatie = nume_locatie.replace(simbol, "-")
            
            # Curatam si punctele sau spatiile care pot incurca caile lungi
            locatie_f = nume_locatie.replace(".", "-").strip()

        # --- C. LOGICA PENTRU CATEGORIE AI ---
        categorie_f = cat_ai if cat_ai else "Diverse"

        # --- D. CONSTRUIRE CALE IERARHICA ---
        # Structura: Destinatie / 2026 / 03 / Lat-45...Lon-28... / Natura
        cale_relativa = os.path.join(an, luna, locatie_f, categorie_f)
        folder_dest_final = os.path.join(destinatie, cale_relativa).replace('\\', '/')
        
        # --- E. CREARE FOLDERE SI COPIERE FIZICA ---
        try:
            # Cream toate folderele deodata (An/Luna/Locatie/Cat)
            os.makedirs(folder_dest_final, exist_ok=True)

            if os.path.exists(cale_orig):
                nume_fisier = os.path.basename(cale_orig)
                cale_fisier_dest = os.path.join(folder_dest_final, nume_fisier).replace('\\', '/')
                
                # Copiem cu copy2 pentru a pastra data crearii originala
                shutil.copy2(cale_orig, cale_fisier_dest)
                contor_succes += 1
            else:
                contor_eroare += 1
        except Exception as e:
            print(f"[EROARE] Problema la fisierul {cale_orig}: {e}")
            contor_eroare += 1
            
        progress.setValue(i + 1)
    
    # 4. DESCHIDERE AUTOMATA FOLDER (Sa vada George unde s-au dus pozele)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(destinatie))
    
    QMessageBox.information(window, "Organizare Finalizata", 
                            f"Procesul s-a incheiat cu succes!\n\n"
                            f"Poze copiate: {contor_succes}\n"
                            f"Erori intampinate: {contor_eroare}")
# --- CAUTARE SI FILTRARE ---

def aplic_filtrare_simpla(text):
    proxy_model.setFilterFixedString(text)
    proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

def executa_cautare_semantic_si_afiseaza(text_cautat):
    """Motorul de cautare FAISS."""
    if index_faiss.ntotal == 0: return
    
    print(f"[AI] Cautare semantica pentru: {text_cautat}")
    v_query = model_ai.encode([f"a photo of {text_cautat}"], normalize_embeddings=True).astype('float32')
    faiss.normalize_L2(v_query)
    
    k = min(40, index_faiss.ntotal)
    distante, indexuri = index_faiss.search(v_query, k)
    
    nume_gasite = []
    for i, idx in enumerate(indexuri[0]):
        # Prag de relevanta (0.20 - 0.23 e optim pentru CLIP)
        if idx != -1 and distante[0][i] > 0.21:
            nume_gasite.append(QtCore.QRegularExpression.escape(os.path.basename(mapare_cai[idx])))
    
    if nume_gasite:
        pattern = "^(" + "|".join(nume_gasite) + ")$"
        opts = QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption
        proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(pattern, opts))
        window.statusBar().showMessage(f"Am gasit {len(nume_gasite)} rezultate pentru '{text_cautat}'")
    else:
        proxy_model.setFilterFixedString("___NIMIC_GASIT___")

def execut_cautare_ai():
    """Ruleaza cautarea FAISS cand apesi Enter."""
    text = search_bar.text().strip()
    if not text:
        proxy_model.setFilterFixedString("")
        afiseaza_toata_libraria()
        return
    
    # Trimitem termenul catre motorul de cautare semantic
    executa_cautare_semantic_si_afiseaza(text)
    
    v_query = model_ai.encode([f"a photo of {text}"], normalize_embeddings=True).astype('float32')
    faiss.normalize_L2(v_query)
    
    k = min(40, index_faiss.ntotal)
    if k == 0: return
    
    dist, idxs = index_faiss.search(v_query, k)
    nume_gasite = []
    for i, idx in enumerate(idxs[0]):
        if idx != -1 and dist[0][i] > 0.20:
            nume_gasite.append(QtCore.QRegularExpression.escape(os.path.basename(mapare_cai[idx])))
    
    if nume_gasite:
        pattern = "^(" + "|".join(nume_gasite) + ")$"
        opts = QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption
        proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(pattern, opts))
    else:
        proxy_model.setFilterFixedString("___NONE___")

# --- CLICK PE TREE: Gestionare Categorii vs Cautari Smart ---
def cand_apas_pe_smart_album_tree(item, col):
    data = item.data(0, Qt.UserRole)
    if not data: return
    
    proxy_model.setFilterFixedString("") # Resetam filtrele vechi
    
    if data.startswith("SEARCH:"):
        # Daca e un Smart Album creat de user, facem cautare AI
        termen = data.replace("SEARCH:", "")
        executa_cautare_semantic_si_afiseaza(termen)
    else:
        # Daca e o categorie standard (Natura, Oameni), filtram din DB
        cai = db_manager.obtine_cai_dupa_categorie(data)
        populeaza_galeria_cu_cai(cai)

def creeaza_album_inteligent():
    """Adauga o cautare salvata direct in smartTreeWidget."""
    nume, ok = QtWidgets.QInputDialog.getText(window, "Album Nou", "Ce cautam semantic (ex: pisici albe):")
    if ok and nume:
        tree = window.findChild(QtWidgets.QTreeWidget, "smartTreeWidget")
        if tree:
            # Cream un item nou in arbore (la nivelul de sus)
            item = QtWidgets.QTreeWidgetItem([f"* {nume}"])
            # Salvam un prefix special in UserRole ca sa stim ca e cautare AI, nu categorie fixa
            item.setData(0, Qt.UserRole, f"SEARCH:{nume}")
            
            # Ii punem un stil italic sa il deosebim de categoriile standard
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
            
            tree.addTopLevelItem(item)
            print(f"[UI] Album Smart adaugat in Tree: {nume}")

# --- FUNCTII ASISTENTA UI ---

def populeaza_galeria_cu_cai(cai):
    model_galerie.clear()
    cai_u = list(dict.fromkeys([os.path.normpath(x).replace('\\', '/') for x in cai]))
    for c in cai_u:
        if not os.path.exists(c): continue
        item = QStandardItem(os.path.basename(c))
        d = db_manager.cauta_dupa_cale(c)
        icon_path = d[10] if (d and d[10] and os.path.exists(d[10])) else c
        item.setData(QIcon(icon_path), Qt.ItemDataRole.DecorationRole)
        item.setData(c, Qt.ItemDataRole.UserRole)
        model_galerie.appendRow(item)

def cand_selectez_o_imagine(index):
    """Citeste toate datele din DB si le trimite catre panoul din dreapta."""
    global procesor_activ
    # Mapam indexul filtrat la cel real
    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    if not cale_fisier: return
    
    # Cautam in baza de date
    date_db = db_manager.cauta_dupa_cale(cale_fisier)
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    
    if date_db:
        # Extragem datele conform structurii tabelului tau:
        # 1:cale, 2:nume, 4:rezolutie, 5:mb, 6:marca, 7:model, 8:data_poza, 9:gps, 10:cale_cache
        info_complete = {
            'nume': date_db[2],
            'rezolutie': date_db[4],
            'mb': date_db[5],
            'marca': date_db[6],
            'model': date_db[7],
            'data': date_db[8],
            'gps': date_db[9]
        }
        
        # Daca avem thumbnail in cache, il folosim
        pixmap = None
        if date_db[10] and os.path.exists(date_db[10]):
            pixmap = QtGui.QPixmap(date_db[10])
        else:
            pixmap = QtGui.QPixmap(cale_fisier) # Backup pe poza originala
            
        actualizeaza_panou_dreapta(info_complete, pixmap)
    else:
        # Daca poza nu e in DB, folosim procesorul worker pentru preview rapid
        if procesor_activ and procesor_activ.isRunning(): 
            procesor_activ.terminate()
        procesor_activ = ProcesorImagine(cale_fisier, preview_label.size())
        procesor_activ.gata_procesarea.connect(actualizeaza_panou_dreapta)
        procesor_activ.start()

def actualizeaza_panou_dreapta(date, pixmap):
    """Afiseaza poza si toate metadatele (Marca, Model, Data, GPS)."""
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    info_label = window.findChild(QtWidgets.QLabel, "infoLabel")
    
    if not preview_label or not info_label: return

    # 1. Afisare Imagine
    if pixmap and not pixmap.isNull():
        pixmap_scalat = pixmap.scaled(preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        preview_label.setPixmap(pixmap_scalat)

    # 2. Constructie Text Metadate (Fara Diacritice)
    detalii = []
    detalii.append(f"<b>Fisier:</b> {date.get('nume', '---')}")
    detalii.append(f"<b>Marime:</b> {date.get('mb', '0')} MB")
    detalii.append(f"<b>Rezolutie:</b> {date.get('rezolutie', '---')}")
    
    # Adaugam datele de camera daca exista
    marca = date.get('marca')
    model = date.get('model')
    if marca and marca != "Necunoscut":
        detalii.append(f"<b>Echipament:</b> {marca} {model if model else ''}")
    
    # Adaugam Data Calendaristica
    data_p = date.get('data')
    if data_p and data_p != "Data Necunoscuta":
        detalii.append(f"<b>Data:</b> {data_p}")
        
    # Adaugam GPS
    gps_p = date.get('gps')
    if gps_p and gps_p != "Fara GPS":
        detalii.append(f"<b>Locatie (GPS):</b> {gps_p}")

    # Punem totul in label cu WordWrap
    info_label.setText("<br>".join(detalii))
    info_label.setWordWrap(True)

def sterge_imaginea_selectata():
    idx = view_galerie.currentIndex()
    if not idx.isValid(): return
    idx_s = proxy_model.mapToSource(idx)
    cale = idx_s.data(Qt.ItemDataRole.UserRole)
    if QMessageBox.question(window, "Stergere", "Stergi imaginea din baza de date?") == QMessageBox.Yes:
        db_manager.sterge_imagine_dupa_cale(cale)
        model_galerie.removeRow(idx_s.row())
        incarca_index_faiss()

def deschide_poza_nativ(index):
    cale = proxy_model.mapToSource(index).data(Qt.ItemDataRole.UserRole)
    if cale and os.path.exists(cale):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(cale))

def arata_meniu_poza(poz):
    idx = view_galerie.indexAt(poz)
    if not idx.isValid(): return
    m = QtWidgets.QMenu(); act = m.addAction("Gaseste poze similare (AI)")
    if m.exec(view_galerie.mapToGlobal(poz)) == act: executa_cautare_similara(idx)

def executa_cautare_similara(index):
    idx_s = proxy_model.mapToSource(index)
    cale = idx_s.data(Qt.ItemDataRole.UserRole)
    date = db_manager.cauta_dupa_cale(cale)
    if not date or not date[12]: return
    v_q = np.array(pickle.loads(date[12])).astype('float32').reshape(1, -1)
    faiss.normalize_L2(v_q)
    dist, idxs = index_faiss.search(v_q, min(10, index_faiss.ntotal))
    nume = [QtCore.QRegularExpression.escape(os.path.basename(mapare_cai[x])) for x in idxs[0] if x != -1]
    if nume:
        ptrn = "^(" + "|".join(nume) + ")$"
        opts = QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption
        proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(ptrn, opts))

def updateaza_status_progres(c, t):
    if window.statusBar(): window.statusBar().showMessage(f"AI: {c}/{t}")

def actualizeaza_iconita_live(a):
    idx = model_galerie.index(a - 1, 0)
    if idx.isValid():
        d = db_manager.cauta_dupa_cale(idx.data(Qt.ItemDataRole.UserRole))
        if d and d[10]: model_galerie.setData(idx, QIcon(d[10]), Qt.ItemDataRole.DecorationRole)

# --- LANSAREA ---
app = QtWidgets.QApplication(sys.argv)
loader = QUiLoader()
window = loader.load("interfata.ui", None)

incarca_index_faiss()

view_galerie = window.findChild(QtWidgets.QListView, "photoView")
model_galerie = QStandardItemModel()
proxy_model = QSortFilterProxyModel()
proxy_model.setSourceModel(model_galerie)

# ASPECT GALERIE
view_galerie.setModel(proxy_model)
view_galerie.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
view_galerie.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
view_galerie.setMovement(QtWidgets.QListView.Movement.Static)
view_galerie.setSpacing(10)
view_galerie.setIconSize(QSize(130, 130))
view_galerie.setGridSize(QSize(160, 180))

# CONEXIUNI EVENIMENTE
view_galerie.clicked.connect(cand_selectez_o_imagine)
view_galerie.doubleClicked.connect(deschide_poza_nativ)
view_galerie.setContextMenuPolicy(Qt.CustomContextMenu)
view_galerie.customContextMenuRequested.connect(arata_meniu_poza)

tree = window.findChild(QtWidgets.QTreeWidget, "smartTreeWidget")
if tree: tree.itemClicked.connect(cand_apas_pe_smart_album_tree)

lista_surse = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
if lista_surse:
    lista_surse.itemClicked.connect(cand_apas_pe_sursa)
    incarca_sursele_vizual()

# CONECTARE BUTOANE
window.findChild(QtWidgets.QPushButton, "btnImportFolder").clicked.connect(adauga_sursa_noua)
window.findChild(QtWidgets.QPushButton, "btnRemoveFolder").clicked.connect(sterge_sursa_selectata)
window.findChild(QtWidgets.QPushButton, "organizeBttn").clicked.connect(executa_organizarea_fizica)
window.findChild(QtWidgets.QPushButton, "bttnAddSmartAlbum").clicked.connect(creeaza_album_inteligent)

search_bar = window.findChild(QtWidgets.QLineEdit, "searchBar")
search_bar.textChanged.connect(aplic_filtrare_simpla)
search_bar.returnPressed.connect(execut_cautare_ai)

shortcut_del = QShortcut(QKeySequence(Qt.Key_Delete), window)
shortcut_del.activated.connect(sterge_imaginea_selectata)

# STARTUP
actualizeaza_smart_albums()
QtCore.QTimer.singleShot(500, afiseaza_toata_libraria)

window.show()
sys.exit(app.exec())