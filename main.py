import sys
import os
import pickle
import numpy as np
import faiss
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon
from PySide6.QtCore import Qt, QSize, QDir, QStandardPaths, QSortFilterProxyModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QShortcut, QKeySequence

import shutil
from PySide6.QtWidgets import QProgressDialog, QMessageBox, QFileDialog

from database import ManagerBazaDate
from scanner_worker import ScannerWorker
from worker import ProcesorImagine
from sentence_transformers import SentenceTransformer

NumeAplicatie = "GalerieLicentaAI"

# --- INITIALIZARE VARIABILE GLOBALE ---
# Acestea previn eroarea NameError: name 'scanner_activ' is not defined
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
        vectori_lista.append(v_numpy)
        mapare_cai.append(cale)
    if vectori_lista:
        v_final = np.array(vectori_lista).astype('float32')
        index_faiss.add(v_final)

def actualizeaza_smart_albums():
    smart_list = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")
    if not smart_list: return
    categorii = ["Oameni", "Natura", "Tehnologie", "Documente", "Arhitectura", "Vehicule", "Animale", "Mancare", "Evenimente", "Diverse"]
    smart_list.clear() 
    for cat in categorii:
        total = db_manager.numara_per_categorie(cat)
        item = QtWidgets.QListWidgetItem(f"{cat} ({total})")
        if total > 0:
            font = item.font(); font.setBold(True); item.setFont(font)
        smart_list.addItem(item)

# --- FUNCTII LOGICA DE LIBRARIE ---

def incarca_sursele_vizual():
    """Populeaza lista din stanga cu folderele adaugate anterior."""
    lista_surse_ui = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
    if not lista_surse_ui: return
    lista_surse_ui.clear()
    
    surse = db_manager.obtine_surse() 
    for s in surse:
        item = QtWidgets.QListWidgetItem(s)
        item.setIcon(window.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        lista_surse_ui.addItem(item)

def adauga_sursa_noua():
    """Deschide dialogul si verifica bifa de recursivitate."""
    cale = QtWidgets.QFileDialog.getExistingDirectory(window, "Selecteaza folderul pentru Librarie")
    
    if cale:
        cale = cale.replace('\\', '/')
        db_manager.adauga_sursa(cale)
        incarca_sursele_vizual()
        
        # --- AICI CITIM STAREA CHECKBOX-ULUI ---
        este_recursiv = False
        check_box = window.findChild(QtWidgets.QCheckBox, "checkRecursive") # Numele din Qt Designer
        if check_box:
            este_recursiv = check_box.isChecked() # Vedem daca e bifat
            
        # Trimitem bifa catre functia de pornire
        porneste_scanare_folder(cale, este_recursiv)

def sterge_sursa_selectata():
    """Sterge sursa selectata din lista si din baza de date."""
    lista_surse_ui = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
    item_selectat = lista_surse_ui.currentItem()
    
    if not item_selectat:
        QtWidgets.QMessageBox.warning(window, "Atentie", "Selecteaza un folder din lista pentru a-l sterge!")
        return
        
    cale_folder = item_selectat.text()
    
    # Confirmare de la utilizator
    raspuns = QtWidgets.QMessageBox.question(
        window, "Stergere Sursa", 
        f"Esti sigur ca vrei sa elimini {cale_folder} din librarie?\nPozele nu vor fi sterse de pe disc.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
    )
    
    if raspuns == QtWidgets.QMessageBox.Yes:
        # 1. Stergem din DB (tabelul surse SI tabelul imagini)
        db_manager.sterge_sursa_si_imagini(cale_folder)
        # 2. Refresh UI
        incarca_sursele_vizual()
        afiseaza_toata_libraria()
        actualizeaza_smart_albums()

def afiseaza_toata_libraria():
    """Arata absolut toate pozele din baza de date."""
    global vizualizare_activa
    vizualizare_activa = "librarie"
    toate_caile = db_manager.obtine_toate_caile_existente()
    populeaza_galeria_cu_cai(toate_caile)
    if window.statusBar():
        window.statusBar().showMessage(f"Librarie totala: {len(toate_caile)} imagini.")

def cand_apas_pe_sursa(item):
    """Filtreaza galeria pentru a arata doar pozele dintr-o sursa specifica."""
    global vizualizare_activa
    vizualizare_activa = "folder"
    cale_folder = item.text()
    
    toate_caile = db_manager.obtine_toate_caile_existente()
    filtrate = [p for p in toate_caile if p.startswith(cale_folder)]
    
    populeaza_galeria_cu_cai(filtrate)
    if window.statusBar():
        window.statusBar().showMessage(f"Folder curent: {len(filtrate)} imagini.")

def porneste_scanare_folder(cale, recursiv=False):
    global scanner_activ
    if scanner_activ and scanner_activ.isRunning():
        scanner_activ.stop(); scanner_activ.wait()
    
    # Trimitem si bifa de recursivitate catre scanner
    scanner_activ = ScannerWorker(cale, folder_cache, cale_db_finala, recursiv)
    scanner_activ.imagine_reparata.connect(actualizeaza_iconita_live)
    scanner_activ.progres.connect(updateaza_status_progres)
    scanner_activ.finalizat.connect(incarca_index_faiss) 
    scanner_activ.finalizat.connect(actualizeaza_smart_albums)
    scanner_activ.finalizat.connect(afiseaza_toata_libraria)
    scanner_activ.start()

def executa_organizarea_fizica():
    # 1. Intrebam utilizatorul UNDE vrea sa creeze noua structura
    director_destinatie = QFileDialog.getExistingDirectory(window, "Alege folderul unde se va crea galeria organizata")
    
    if not director_destinatie:
        return # Utilizatorul a apasat Cancel

    # 2. Luam datele din baza de date
    date_poze = db_manager.obtine_toate_pentru_organizare()
    if not date_poze:
        QMessageBox.warning(window, "Atentie", "Nu exista poze scanate in baza de date!")
        return

    # 3. Pregatim bara de progres
    progress = QProgressDialog("Se organizeaza pozele...", "Anuleaza", 0, len(date_poze), window)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setWindowTitle("Organizare AI")
    progress.show()

    contor = 0
    for cale_orig, categorie in date_poze:
        if progress.wasCanceled():
            break
            
        # Daca AI-ul nu a fost sigur, punem poza in "Diverse"
        cat_folder = categorie if categorie else "Diverse"
        
        # Cream folderul categoriei (ex: D:/Export/Natura)
        cale_folder_nou = os.path.join(director_destinatie, cat_folder).replace('\\', '/')
        if not os.path.exists(cale_folder_nou):
            os.makedirs(cale_folder_nou)

        # Pregatim calea finala a fisierului
        nume_fisier = os.path.basename(cale_orig)
        cale_finala_fisier = os.path.join(cale_folder_nou, nume_fisier).replace('\\', '/')

        # Copiem fizic fisierul (copy2 pastreaza data si metadatele)
        try:
            if os.path.exists(cale_orig):
                shutil.copy2(cale_orig, cale_finala_fisier)
        except Exception as e:
            print(f"Eroare la copierea fisierului {nume_fisier}: {e}")

        contor += 1
        progress.setValue(contor)

    progress.setValue(len(date_poze))
    QMessageBox.information(window, "Succes", f"Organizarea a fost finalizata! {contor} poze au fost copiate.")

def deschide_poza_nativ(index):
        """Deschide imaginea in programul implicit de vizualizare din Windows."""
        # Mapam indexul de la Proxy (filtru) la Modelul de baza
        index_sursa = proxy_model.mapToSource(index)
        cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
        
        if cale_fisier and os.path.exists(cale_fisier):
            # QDesktopServices stie sa vorbeasca direct cu Windows-ul
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(cale_fisier))


def sterge_imaginea_selectata():
    """Sterge imaginea selectata curent din DB si din Galerie."""
    
    print("Scurtatura Delete a fost activata!") # <--- Linie de test
    index = view_galerie.currentIndex()
    if not index.isValid(): 
        print("Nicio imagine selectata!")
        return
    

    index = view_galerie.currentIndex()
    if not index.isValid(): return

    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    nume = os.path.basename(cale_fisier)

    # Confirmare rapida
    raspuns = QMessageBox.question(window, "Confirmare Stergere", 
                                  f"Sigur vrei sa scoti poza '{nume}' din galerie?\n(Fisierul ramane pe disc)")
    
    if raspuns == QMessageBox.Yes:
        # 1. Stergem din DB
        db_manager.sterge_imagine_dupa_cale(cale_fisier)
        # 2. Stergem din Model (dispare vizual)
        model_galerie.removeRow(index_sursa.row())
        window.statusBar().showMessage(f"Poza {nume} a fost eliminata.", 3000)

# --- SUPRASCRIERE EVENIMENT TASTE ---
def keyPressEvent(event):
    if event.key() == Qt.Key_Delete:
        sterge_imaginea_selectata()
    # Poti adauga si alte taste aici, ex: tastele sageti pentru navigare
    # Ii spunem ferestrei sa foloseasca functia noastra de ascultare
    window.keyPressEvent = keyPressEvent


# --- FUNCTII AFISARE GALERIE ---

def populeaza_galeria_cu_cai(cai_fisiere):
    model_galerie.clear()
    cai_unice = list(dict.fromkeys([os.path.normpath(c).replace('\\', '/') for c in cai_fisiere]))
    for cale_full in cai_unice:
        if not os.path.exists(cale_full): continue
        item = QStandardItem(os.path.basename(cale_full))
        date_db = db_manager.cauta_dupa_cale(cale_full)
        cale_iconita = cale_full 
        if date_db and len(date_db) > 10 and date_db[10] and os.path.exists(date_db[10]):
            cale_iconita = date_db[10]
        item.setData(QIcon(cale_iconita), Qt.ItemDataRole.DecorationRole)
        item.setData(cale_full, Qt.ItemDataRole.UserRole)
        model_galerie.appendRow(item)

def cand_selectez_o_imagine(index):
    global procesor_activ
    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    if not cale_fisier: return
    
    cale_pt_db = cale_fisier.replace('\\', '/')
    date_db = db_manager.cauta_dupa_cale(cale_pt_db)
    
    if date_db and date_db[10] and os.path.exists(date_db[10]):
        info_d = {
            'cale': date_db[1], 'nume': date_db[2], 'rezolutie': date_db[4], 
            'mb': date_db[5], 'marca': date_db[6], 'model': date_db[7], 'gps': date_db[9]
        }
        actualizeaza_panou_dreapta(info_d, QtGui.QPixmap(date_db[10]))
        return
        
    if procesor_activ and procesor_activ.isRunning(): 
        procesor_activ.terminate()
        
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    procesor_activ = ProcesorImagine(cale_fisier, preview_label.size())
    procesor_activ.gata_procesarea.connect(actualizeaza_panou_dreapta)
    procesor_activ.start()

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

def updateaza_status_progres(curent, total):
    if window.statusBar(): window.statusBar().showMessage(f"AI Analiza: {curent}/{total}")

def actualizeaza_iconita_live(actual):
    index_model = model_galerie.index(actual - 1, 0)
    if not index_model.isValid(): return
    cale_originala = index_model.data(Qt.ItemDataRole.UserRole)
    date = db_manager.cauta_dupa_cale(cale_originala)
    if date and len(date) > 10 and date[10] and os.path.exists(date[10]):
        model_galerie.setData(index_model, QIcon(date[10]), Qt.ItemDataRole.DecorationRole)

def aplic_filtrare_simpla(text):
    proxy_model.setFilterFixedString(text)
    proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

def execut_cautare_ai():
    text_original = search_bar.text().strip()
    if not text_original:
        proxy_model.setFilterFixedString(""); return
    text_optimizat = f"a photo of {text_original}"
    vector_text = model_ai.encode([text_optimizat], normalize_embeddings=True).astype('float32')
    k_cerut = min(5, index_faiss.ntotal)
    if k_cerut == 0: return
    distante, indexuri = index_faiss.search(vector_text, k_cerut)
    prag_relevanta = 0.21 
    nume_gasite = []
    for i, idx in enumerate(indexuri[0]):
        if idx != -1 and distante[0][i] > prag_relevanta:
            nume_gasite.append(QtCore.QRegularExpression.escape(os.path.basename(mapare_cai[idx])))
    if nume_gasite:
        pattern = "^(" + "|".join(nume_gasite) + ")$"
        proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.CaseInsensitiveOption))
    else:
        proxy_model.setFilterFixedString("___NIMIC_GASIT___")

def cand_apas_pe_smart_album(item):
    global vizualizare_activa
    vizualizare_activa = "smart"
    text_complet = item.text()
    proxy_model.setFilterFixedString("") 
    if text_complet.startswith("*"):
        termen = text_complet.replace("* ", "")
        executa_cautare_semantic_si_afiseaza(termen)
    else:
        categorie = text_complet.split(" (")[0]
        cai_fisiere = db_manager.obtine_cai_dupa_categorie(categorie)
        if not cai_fisiere: model_galerie.clear(); return
        populeaza_galeria_cu_cai(cai_fisiere)

def executa_cautare_semantic_si_afiseaza(text_cautat):
    if index_faiss is None or index_faiss.ntotal == 0: return
    prompt_en = f"a photo of {text_cautat}"
    vector_cautare = model_ai.encode([prompt_en], normalize_embeddings=True).astype('float32')
    k = min(40, index_faiss.ntotal)
    distante, indexuri = index_faiss.search(vector_cautare, k)
    cai_gasite = []
    prag_relevanta = 0.23 
    for i, idx in enumerate(indexuri[0]):
        if idx != -1 and distante[0][i] > prag_relevanta:
            cai_gasite.append(mapare_cai[idx])
    populeaza_galeria_cu_cai(cai_gasite)

def creeaza_album_inteligent():
    prompt, ok = QtWidgets.QInputDialog.getText(window, "Colectie noua", "Ce vrei sa contina acest album?")
    if ok and prompt:
        smart_list = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")
        if smart_list:
            item = QtWidgets.QListWidgetItem(f"* {prompt}")
            font = item.font(); font.setItalic(True); item.setFont(font)
            smart_list.addItem(item); smart_list.setCurrentItem(item)
            cand_apas_pe_smart_album(item)

def arata_meniu_poza(pozitie):
    index = view_galerie.indexAt(pozitie)
    if not index.isValid(): return
    
    meniu = QtWidgets.QMenu()
    actiune_similare = meniu.addAction("Gaseste poze similare (AI)")
    
    actiune = meniu.exec(view_galerie.mapToGlobal(pozitie))
    
    if actiune == actiune_similare:
        executa_cautare_similara(index)

def executa_cautare_similara(index):
    index_sursa = proxy_model.mapToSource(index)
    cale_poza = index_sursa.data(Qt.ItemDataRole.UserRole)
    
    # 1. Luam vectorul pozei din DB
    date_poza = db_manager.cauta_dupa_cale(cale_poza)
    if not date_poza or date_poza[12] is None: 
        print("Nu am vector AI pentru aceasta poza!"); return
    
   # In loc de np.frombuffer, folosim pickle.loads pentru ca asa e salvat in DB
    try:
        vector_raw = pickle.loads(date_poza[12])
        # Ne asiguram ca e un array numpy de tip float32 si are forma (1, 512)
        vector_query = np.array(vector_raw).astype('float32').reshape(1, -1)
    except Exception as e:
        print(f"Eroare la decodarea vectorului: {e}")
        return
    # -----------------------
    
    # 2. Cautam in FAISS cele mai apropiate 10 rezultate
    k = min(10, index_faiss.ntotal)
    if k == 0: return
    print(f"DEBUG: Dimensiune index FAISS: {index_faiss.d}")
    print(f"DEBUG: Dimensiune vector query: {vector_query.shape}")
    distante, indexuri = index_faiss.search(vector_query, k)
    
    # 3. Filtram galeria sa arate doar acele poze
    nume_gasite = []
    for i, idx in enumerate(indexuri[0]):
        if idx != -1: # FAISS returneaza -1 daca nu gaseste destule
            nume_gasite.append(QtCore.QRegularExpression.escape(os.path.basename(mapare_cai[idx])))
    
    if nume_gasite:
        pattern = "^(" + "|".join(nume_gasite) + ")$"
        proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.CaseInsensitiveOption))
        window.statusBar().showMessage(f"Am gasit {len(nume_gasite)} poze similare.")

def arata_meniu_folder(pozitie):
    item = lista_surse.itemAt(pozitie)
    if not item: return
    
    meniu = QtWidgets.QMenu()
    actiune_scan = meniu.addAction("Rescanare (Cauta poze noi)")
    
    # Executam meniul si vedem ce a ales userul
    actiune = meniu.exec(lista_surse.mapToGlobal(pozitie))
    check_box = window.findChild(QtWidgets.QCheckBox, "checkRecursive") 
    if actiune == actiune_scan:
        cale_folder = item.text()
        recursiv = check_box.isChecked() if check_box else False
        print(f"[Sistem] Rescanare folder: {cale_folder}")
        porneste_scanare_folder(cale_folder, recursiv)

# --- LANSAREA ---
app = QtWidgets.QApplication(sys.argv)
loader = QUiLoader()
window = loader.load("interfata.ui", None)

incarca_index_faiss()

view_galerie = window.findChild(QtWidgets.QListView, "photoView")
model_galerie = QStandardItemModel()
proxy_model = QSortFilterProxyModel()
proxy_model.setSourceModel(model_galerie)

view_galerie.setModel(proxy_model)
view_galerie.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
view_galerie.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
view_galerie.setMovement(QtWidgets.QListView.Movement.Static)
view_galerie.setSpacing(10)
view_galerie.setIconSize(QSize(130, 130))
view_galerie.setGridSize(QSize(160, 180))
view_galerie.clicked.connect(cand_selectez_o_imagine)

# --- CONFIGURARE LISTA SURSE ---
lista_surse = window.findChild(QtWidgets.QListWidget, "sourceListWidget")
if lista_surse:
    lista_surse.itemClicked.connect(cand_apas_pe_sursa)
    incarca_sursele_vizual()

# Permitem meniuri personalizate
lista_surse.setContextMenuPolicy(Qt.CustomContextMenu)
lista_surse.customContextMenuRequested.connect(arata_meniu_folder)

view_galerie.setContextMenuPolicy(Qt.CustomContextMenu)
view_galerie.customContextMenuRequested.connect(arata_meniu_poza)

btn_import = window.findChild(QtWidgets.QPushButton, "btnImportFolder")
if btn_import:
    btn_import.clicked.connect(adauga_sursa_noua)

btn_remove = window.findChild(QtWidgets.QPushButton, "btnRemoveFolder") 
if btn_remove:
    btn_remove.clicked.connect(sterge_sursa_selectata)

search_bar = window.findChild(QtWidgets.QLineEdit, "searchBar")
if search_bar:
    search_bar.textChanged.connect(aplic_filtrare_simpla)
    search_bar.returnPressed.connect(execut_cautare_ai)

smart_album_view = window.findChild(QtWidgets.QListWidget, "smartAlbumWidget")
if smart_album_view:
    smart_album_view.itemClicked.connect(cand_apas_pe_smart_album)

btn_colectie_noua = window.findChild(QtWidgets.QPushButton, "bttnAddSmartAlbum")
if btn_colectie_noua:
    btn_colectie_noua.clicked.connect(creeaza_album_inteligent)

btn_organize = window.findChild(QtWidgets.QPushButton, "organizeBttn")
if btn_organize:
    btn_organize.clicked.connect(executa_organizarea_fizica)

view_galerie.doubleClicked.connect(deschide_poza_nativ)

#Cream scurtatura pentru tasta Delete
shortcut_delete = QShortcut(QKeySequence(Qt.Key_Delete), window)

#O conectam la functia de stergere
shortcut_delete.activated.connect(sterge_imaginea_selectata)



actualizeaza_smart_albums()

# LA START: Incarcam totul din DB
QtCore.QTimer.singleShot(500, afiseaza_toata_libraria)

window.show()
sys.exit(app.exec())