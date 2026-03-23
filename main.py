import sys
import os
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon, QPixmap
from PySide6.QtCore import Qt, QSize, QDir, QStandardPaths, QSortFilterProxyModel
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QFileSystemModel

# === IMPORTAM CLASA DIN CELALALT FISIER ===
from worker import ProcesorImagine 

# --- FUNCTIILE DE INTERFATA ---

def actualizeaza_panou_dreapta(date, pixmap):
    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    info_label = window.findChild(QtWidgets.QLabel, "infoLabel")
    
    if not pixmap.isNull():
        preview_label.setPixmap(pixmap)
    
    linii = [
        f"<b>Fisier:</b> {date.get('nume', '---')}",
        f"<b>Format:</b> {date.get('format', '---')}",
        f"<b>Rezolutie:</b> {date.get('rezolutie', '---')}",
        f"<b>Marime:</b> {date.get('mb', 0)} MB"
    ]
    if 'marca' in date: linii.append(f"<b>Marca:</b> {date['marca']}")
    if 'model' in date: linii.append(f"<b>Model:</b> {date['model']}")
    if 'data' in date: linii.append(f"<b>Data:</b> {date['data']}")
    if 'gps' in date: linii.append(f"<b>GPS:</b> {date['gps']}")
    
    linii.append(f"<br><b>Cale:</b><br><small>{date.get('cale', '')}</small>")
    info_label.setText("<br>".join(linii))
    info_label.setWordWrap(True)

procesor_activ = None

def cand_selectez_o_imagine(index):
    global procesor_activ
    index_sursa = proxy_model.mapToSource(index)
    cale_fisier = index_sursa.data(Qt.ItemDataRole.UserRole)
    if not cale_fisier: return

    if procesor_activ is not None:
        try:
            if procesor_activ.isRunning():
                procesor_activ.terminate()
                procesor_activ.wait()
        except RuntimeError:
            procesor_activ = None

    preview_label = window.findChild(QtWidgets.QLabel, "previewLabel")
    info_label = window.findChild(QtWidgets.QLabel, "infoLabel")
    if info_label: info_label.setText("<i>Se repara orientarea...</i>")

    procesor_activ = ProcesorImagine(cale_fisier, preview_label.size())
    procesor_activ.gata_procesarea.connect(actualizeaza_panou_dreapta)
    procesor_activ.finished.connect(procesor_activ.deleteLater)
    procesor_activ.start()

def cand_apas_pe_folder(index):
    cale_folder = tree_model.filePath(index)
    if not os.path.isdir(cale_folder): cale_folder = os.path.dirname(cale_folder)
    model_galerie.clear()
    formate = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    try:
        for nume in os.listdir(cale_folder):
            if nume.lower().endswith(formate):
                cale_full = os.path.join(cale_folder, nume)
                item = QStandardItem(nume)
                item.setData(QIcon(cale_full), Qt.ItemDataRole.DecorationRole)
                item.setData(cale_full, Qt.ItemDataRole.UserRole)
                model_galerie.appendRow(item)
    except Exception as e: print(f"Eroare: {e}")

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