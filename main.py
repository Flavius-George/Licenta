import sys
import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap
from PySide6.QtCore import Qt, QSize, QDir, QStandardPaths, QSortFilterProxyModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileSystemModel
from database import ManagerBazaDate
from worker import ProcesorImagine
from scanner_worker import ScannerWorker

#SE CREEAZA FISIERUL DE BAZE DE DATE
db_manager = ManagerBazaDate()
# === IMPORTAM CLASA DIN CELALALT FISIER ===
from worker import ProcesorImagine 

# --- FUNCTIILE DE INTERFATA ---

def actualizeaza_panou_dreapta(date, pixmap):
    """
    Afiseaza imaginea scalata corect si toate metadatele extrase.
    Sistem hibrid: daca pixmap e gol, actualizeaza doar textul.
    """
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    info_label = window.findChild(QtWidgets.QLabel, "infoLabel")

    if not preview_label or not info_label:
        return

    # 1. ACTUALIZARE IMAGINE (Doar daca am primit pixeli de la Worker)
    # Daca pixmap.isNull() este True, inseamna ca datele vin din DB si sarim peste asta,
    # pastrand ce era inainte pe label (sau ramanand gol pana vine Worker-ul).
    if pixmap and not pixmap.isNull():
        pixmap_scalat = pixmap.scaled(
            preview_label.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        preview_label.setPixmap(pixmap_scalat)
        preview_label.setAlignment(QtCore.Qt.AlignCenter)

    # 2. CONSTRUIRE TEXT DETALII (Se executa mereu)
    detalii = [
        f"<b>Fisier:</b> {date.get('nume', '---')}",
        f"<b>Format:</b> {date.get('format', '---')}",
        f"<b>Rezolutie:</b> {date.get('rezolutie', '---')}",
        f"<b>Marime:</b> {date.get('mb', 0)} MB"
    ]

    # Adaugam datele EXIF in lista daca exista
    if date.get('marca'):  detalii.append(f"<b>Marca:</b> {date['marca']}")
    if date.get('model'):  detalii.append(f"<b>Model:</b> {date['model']}")
    if date.get('data'):   detalii.append(f"<b>Data:</b> {date['data']}")
    if date.get('gps'):    detalii.append(f"<b>GPS:</b> {date['gps']}")

    # Adaugam calea la final
    detalii.append(f"<br><b>Cale completa:</b><br><small>{date.get('cale', '')}</small>")

    # Aplicam textul pe label
    info_label.setText("<br>".join(detalii))
    info_label.setWordWrap(True)

    # 3. SALVARE/ACTUALIZARE IN DB
    # Salvam doar daca avem un dictionar valid
    if date:
        db_manager.salveaza_sau_actualizeaza(date)

procesor_activ = None
scanner_activ = None

def cand_selectez_o_imagine(index):
    global procesor_activ
    
    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    if not cale_fisier: 
        return

    # --- 1. SMART LOAD (VERIFICARE DB SI CACHE) ---
    date_existente = db_manager.cauta_dupa_cale(cale_fisier)
    
    if date_existente:
        # Reconstruim dictionarul
        date_db = {
            'cale': date_existente[1],
            'nume': date_existente[2],
            'format': date_existente[3],
            'rezolutie': date_existente[4],
            'mb': date_existente[5],
            'marca': date_existente[6],
            'model': date_existente[7],
            'data': date_existente[8],
            'gps': date_existente[9]
        }
        
        # Verificam daca avem imaginea deja reparata in CACHE
        # In DB, cale_cache este pe coloana 10
        cale_cache = date_existente[10] if len(date_existente) > 10 else None
        
        if cale_cache and os.path.exists(cale_cache):
            # DACA AVEM CACHE: Afisam TOTUL instant (imagine + text) si oprim functia
            actualizeaza_panou_dreapta(date_db, QtGui.QPixmap(cale_cache))
            print(f"Incarcare INSTANT din Cache pentru: {date_db['nume']}")
            return 
        else:
            # DACA NU AVEM CACHE INCA: Punem textul si lasam Worker-ul de jos sa lucreze
            actualizeaza_panou_dreapta(date_db, QtGui.QPixmap())
            print(f"Metadate din DB, dar se genereaza imaginea pentru: {date_db['nume']}")

    # --- 2. SIGURANTA THREAD ---
    if procesor_activ is not None:
        try:
            if procesor_activ.isRunning():
                procesor_activ.terminate()
                procesor_activ.wait()
        except RuntimeError:
            procesor_activ = None

    # --- 3. PORNIRE WORKER (Doar daca nu am avut Cache valid) ---
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    
    if not date_existente:
        info_label = window.findChild(QtWidgets.QLabel, "infoLabel")
        if info_label: 
            info_label.setText("<i>Se analizeaza imaginea pentru prima oara...</i>")

    procesor_activ = ProcesorImagine(cale_fisier, preview_label.size())
    procesor_activ.gata_procesarea.connect(actualizeaza_panou_dreapta)
    procesor_activ.finished.connect(procesor_activ.deleteLater)
    procesor_activ.start()

def actualizeaza_iconita_live(actual, total):
    """
    Se executa in timp ce ScannerWorker lucreaza. 
    Inlocuieste iconita 'bruta' cu cea rotita corect din cache.
    """
    if model_galerie.rowCount() == 0: 
        return
    
    # Indexul in model corespunde cu ordinea fisierelor (actual - 1)
    index_model = model_galerie.index(actual - 1, 0)
    if not index_model.isValid(): 
        return
    
    cale_originala = index_model.data(Qt.ItemDataRole.UserRole)
    if not cale_originala: 
        return
    
    # Intreabam baza de date de noua cale de cache
    date = db_manager.cauta_dupa_cale(cale_originala)
    
    if date and len(date) > 10 and date[10]:
        cale_cache = date[10]
        if os.path.exists(cale_cache):
            # Punem iconita noua, rotita corect
            icon_nou = QIcon(cale_cache)
            model_galerie.setData(index_model, icon_nou, Qt.ItemDataRole.DecorationRole)

def cand_apas_pe_folder(index):
    global scanner_activ
    
    # 1. AFLAM CALEA FOLDERULUI
    cale_folder = tree_model.filePath(index)
    if not os.path.isdir(cale_folder): 
        cale_folder = os.path.dirname(cale_folder)
    
    model_galerie.clear()
    formate = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    # 2. AFISARE INITIALA (Sortata si cu Cache verificat)
    try:
        # Preluam lista de fisiere si o SORTAM alfabetic (extrem de important!)
        nume_fisiere = os.listdir(cale_folder)
        fisiere_valide = sorted([f for f in nume_fisiere if f.lower().endswith(formate)])
        
        for nume in fisiere_valide:
            cale_full = os.path.join(cale_folder, nume)
            item = QStandardItem(nume)
            
            # Verificam in DB daca avem deja cache
            date_existente = db_manager.cauta_dupa_cale(cale_full)
            cale_icon = cale_full # Incepem cu originalul
            
            if date_existente and len(date_existente) > 10 and date_existente[10]:
                cale_cache = date_existente[10]
                # Daca fisierul cache chiar exista fizic, il folosim direct
                if os.path.exists(cale_cache):
                    cale_icon = cale_cache 

            item.setData(QIcon(cale_icon), Qt.ItemDataRole.DecorationRole)
            item.setData(cale_full, Qt.ItemDataRole.UserRole)
            model_galerie.appendRow(item)
            
    except Exception as e: 
        print(f"Eroare afisare galerie: {e}")

    # 3. GESTIONARE SCANNER
    if scanner_activ and scanner_activ.isRunning():
        # IMPORTANT: Setam running = False inainte de wait ca sa se opreasca rapid
        scanner_activ.stop()
        scanner_activ.wait()

    # Pornim scanarea noua
    scanner_activ = ScannerWorker(cale_folder)
    
    # Conectam update-ul LIVE
    scanner_activ.progres.connect(actualizeaza_iconita_live)
    
    if window.statusBar():
        scanner_activ.progres.connect(
            lambda cur, tot: window.statusBar().showMessage(f"Indexare folder: {cur}/{tot}")
        )
        scanner_activ.finalizat.connect(
            lambda: window.statusBar().showMessage("Indexare completa.", 3000)
        )
        
    scanner_activ.start()

def aplic_filtrare_search(text):
    proxy_model.setFilterFixedString(text)
    proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

# --- LANSAREA ---
app = QtWidgets.QApplication(sys.argv)
loader = QUiLoader()
window = loader.load("interfata.ui", None)

view_galerie = window.findChild(QtWidgets.QListView, "photoView")
model_galerie = QStandardItemModel()
proxy_model = QSortFilterProxyModel()
proxy_model.setSourceModel(model_galerie)
view_galerie.setModel(proxy_model)
view_galerie.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
view_galerie.setIconSize(QSize(120, 120))
view_galerie.setGridSize(QSize(140, 140))
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

search_bar = window.findChild(QtWidgets.QLineEdit, "searchBar")
if search_bar: search_bar.textChanged.connect(aplic_filtrare_search)

window.show()
sys.exit(app.exec())