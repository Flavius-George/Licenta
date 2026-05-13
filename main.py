import sys
import os
import pickle
import numpy as np
import faiss
import shutil
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QStandardItemModel, QStandardItem, QIcon, QShortcut, QKeySequence
from PySide6.QtCore import Qt, QSize, QStandardPaths, QThread
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QProgressDialog, QMessageBox, QFileDialog
from PySide6.QtCore import QSortFilterProxyModel

from database import ManagerBazaDate
from scanner_worker import ScannerWorker
from worker import ProcesorImagine
from sentence_transformers import SentenceTransformer

# ============================================================
# CONSTANTE GLOBALE
# ============================================================

NUME_APLICATIE = "GalerieLicentaAI"

# Ierarhia vizuala din QTreeWidget
STRUCTURA_ALBUME = {
    "A. Viata Personala": {
        "Evenimente": ["Nunti", "Petreceri", "Sarbatori"],
        "Oameni":     ["Oameni"],
        "Mancare":    ["Mancare"],
    },
    "B. Profesional & Academic": {
        "Documente":  ["Documente Clasice", "Diagrame & Scheme", "Baze de Date"],
        "Tehnologie": ["Hardware", "Interfete Software", "Screenshots Cod"],
    },
    "C. Mediu & Obiecte": {
        "Natura":      ["Natura", "Animale"],
        "Arhitectura": ["Arhitectura"],
        "Vehicule":    ["Vehicule"],
    },
}

# Praguri CLIP (justificate experimental in teza)
PRAG_CAUTARE_SEMANTICA   = 0.21   # pentru cautare text -> imagine
PRAG_SIMILITUDINE_VIZUALA = 0.20  # pentru "gaseste similare"

# Daca True, scanarea foloseste TTA (Test-Time Augmentation) cu 5 crop-uri
# pentru o clasificare mai robusta (recomandat pentru demo-ul de licenta).
# Pune pe False daca vrei scanari de ~5x mai rapide cand testezi pe colectii mari.
SCANARE_CU_TTA = True

# ----------------------------------------------------------
# Coloane tabel SQLite — schema cu geocodare
# ----------------------------------------------------------
COL_CALE       = 1
COL_NUME       = 2
COL_FORMAT     = 3
COL_REZOLUTIE  = 4
COL_MB         = 5
COL_MARCA      = 6
COL_MODEL      = 7
COL_DATA_POZA  = 8
COL_GPS        = 9
COL_LAT        = 10
COL_LON        = 11
COL_ORAS       = 12
COL_TARA       = 13
COL_TARA_COD   = 14
COL_CALE_CACHE = 15
COL_CATEGORIE  = 16
COL_VECTOR_AI  = 17


# ============================================================
# CLASA PRINCIPALA
# ============================================================

class MainWindow:
    """
    Controller principal al aplicatiei GalerieLicentaAI.

    Responsabilitati:
    - Gestioneaza fereastra UI incarcata din interfata.ui
    - Conecteaza semnale/sloturi pentru toate actiunile utilizatorului
    - Coordoneaza modulele: DB, ScannerWorker, ProcesorImagine, FAISS
    """

    def __init__(self):
        # --- Cai sistem ---
        self._initializeaza_cai()

        # --- Baza de date ---
        self.db = ManagerBazaDate(self.cale_db)

        # --- Model AI (CLIP) - ENCARCAT O SINGURA DATA SI PARTAJAT CU SCANNER ---
        print("[AI] Se incarca modelul CLIP (clip-ViT-B-32)...")
        self.model_ai = SentenceTransformer('clip-ViT-B-32')

        # --- Index FAISS (cautare vectoriala rapida) ---
        self.index_faiss = faiss.IndexFlatIP(512)  # 512 = dim vector CLIP
        self.mapare_cai: list[str] = []

        # --- Stare interna ---
        self.scanner_activ: ScannerWorker | None = None
        self.procesor_activ: ProcesorImagine | None = None
        self.vizualizare_activa = "librarie"  # "librarie" | "folder" | "album"

        # --- UI ---
        loader = QUiLoader()
        self.window = loader.load("interfata.ui", None)
        self._initializeaza_galerie()
        self._conecteaza_semnale()

        # --- Startup ---
        self.incarca_index_faiss()
        self.actualizeaza_smart_albums()
        self.incarca_sursele_vizual()
        QtCore.QTimer.singleShot(500, self.afiseaza_toata_libraria)

        self.window.show()

    # ----------------------------------------------------------
    # INITIALIZARE
    # ----------------------------------------------------------

    def _initializeaza_cai(self):
        """Pregateste folderele AppData si Cache."""
        folder_app_data = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        os.makedirs(folder_app_data, exist_ok=True)

        folder_cache_root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation
        )
        self.folder_cache = os.path.join(folder_cache_root, NUME_APLICATIE, "cache").replace("\\", "/")
        os.makedirs(self.folder_cache, exist_ok=True)

        self.cale_db = os.path.join(folder_app_data, "galerie_licenta.db").replace("\\", "/")

    def _initializeaza_galerie(self):
        """Configureaza QListView + modelul de date + proxy de filtrare."""
        self.view_galerie: QtWidgets.QListView = self.window.findChild(
            QtWidgets.QListView, "photoView"
        )
        self.model_galerie = QStandardItemModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model_galerie)

        self.view_galerie.setModel(self.proxy_model)
        self.view_galerie.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.view_galerie.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.view_galerie.setMovement(QtWidgets.QListView.Movement.Static)
        self.view_galerie.setSpacing(10)
        self.view_galerie.setIconSize(QSize(130, 130))
        self.view_galerie.setGridSize(QSize(160, 180))

    def _conecteaza_semnale(self):
        """Conecteaza toate evenimentele UI la metodele controller-ului."""
        w = self.window

        # Galerie
        self.view_galerie.clicked.connect(self.cand_selectez_o_imagine)
        self.view_galerie.doubleClicked.connect(self.deschide_poza_nativ)
        self.view_galerie.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view_galerie.customContextMenuRequested.connect(self.arata_meniu_poza)

        # Tree albume inteligente
        tree = w.findChild(QtWidgets.QTreeWidget, "smartTreeWidget")
        if tree:
            tree.itemClicked.connect(self.cand_apas_pe_smart_album_tree)

        # Lista surse
        lista_surse = w.findChild(QtWidgets.QListWidget, "sourceListWidget")
        if lista_surse:
            lista_surse.itemClicked.connect(self.cand_apas_pe_sursa)

        # Butoane
        w.findChild(QtWidgets.QPushButton, "btnImportFolder").clicked.connect(self.adauga_sursa_noua)
        w.findChild(QtWidgets.QPushButton, "btnRemoveFolder").clicked.connect(self.sterge_sursa_selectata)
        w.findChild(QtWidgets.QPushButton, "organizeBttn").clicked.connect(self.executa_organizarea_fizica)
        w.findChild(QtWidgets.QPushButton, "bttnAddSmartAlbum").clicked.connect(self.creeaza_album_inteligent)

        # Search bar
        self.search_bar: QtWidgets.QLineEdit = w.findChild(QtWidgets.QLineEdit, "searchBar")
        self.search_bar.textChanged.connect(self.aplic_filtrare_simpla)
        self.search_bar.returnPressed.connect(self.execut_cautare_ai)

        # Shortcut DELETE
        QShortcut(QKeySequence(Qt.Key_Delete), self.window).activated.connect(
            self.sterge_imaginea_selectata
        )

    # ----------------------------------------------------------
    # AI & FAISS
    # ----------------------------------------------------------

    def incarca_index_faiss(self):
        """Reconstruieste indexul FAISS din vectorii stocati in SQLite."""
        self.index_faiss.reset()
        self.mapare_cai = []

        date_vectori = self.db.obtine_toti_vectorii()
        if not date_vectori:
            return

        vectori_lista = []
        for cale, v_numpy in date_vectori:
            v = v_numpy.astype("float32")
            faiss.normalize_L2(v.reshape(1, -1))
            vectori_lista.append(v)
            self.mapare_cai.append(cale)

        if vectori_lista:
            self.index_faiss.add(np.array(vectori_lista, dtype="float32"))
            print(f"[FAISS] Index pregatit cu {len(self.mapare_cai)} vectori.")

    def _encode_text_query(self, text: str) -> np.ndarray:
        """Prompt Ensembling: mediaza mai multe variante de text pentru a stabiliza cautarea."""
        templates = [
            f"a photo of {text}",
            f"a close-up photo of {text}",
            f"a blurry photo of {text}",
            f"a high quality photo of {text}",
            f"a picture containing {text}"
        ]
        v_lista = self.model_ai.encode(templates, normalize_embeddings=True)
        v_mediu = np.mean(v_lista, axis=0)
        faiss.normalize_L2(v_mediu.reshape(1, -1))
        return v_mediu.astype("float32")

    def cauta_semantic(self, text: str, k: int = 40, prag: float = 0.21) -> list[str]:
        """Cautare cu prag (0.21) pentru a elimina rezultatele zgomotoase."""
        if self.index_faiss.ntotal == 0:
            return []

        v_query = self._encode_text_query(text)
        k_efectiv = min(k, self.index_faiss.ntotal)
        distante, indexuri = self.index_faiss.search(v_query.reshape(1, -1), k_efectiv)

        cai_gasite = []
        for scor, idx in zip(distante[0], indexuri[0]):
            if idx != -1 and scor > prag:
                cai_gasite.append(self.mapare_cai[idx])
        return cai_gasite

    # ----------------------------------------------------------
    # SCANARE
    # ----------------------------------------------------------

    def porneste_scanare_folder(self, cale: str, recursiv: bool = False):
        """Porneste ScannerWorker pentru un folder nou."""
        if self.scanner_activ and self.scanner_activ.isRunning():
            self.scanner_activ.stop()
            self.scanner_activ.wait()

        # ---- PARTAJAM MODELUL CLIP cu scanner-ul ----
        # Astfel evitam o a doua copie de ~600MB in RAM si timpul de
        # reincarcare (~5-15 secunde).
        self.scanner_activ = ScannerWorker(
            cale,
            self.folder_cache,
            self.cale_db,
            recursiv,
            model=self.model_ai,
            cu_tta=SCANARE_CU_TTA,
        )
        self.scanner_activ.imagine_reparata.connect(self._actualizeaza_iconita_live)
        self.scanner_activ.progres.connect(self._updateaza_status_progres)
        self.scanner_activ.finalizat.connect(self._dupa_scanare_finalizata)

        # ---- PRIORITATE SCAZUTA pentru thread-ul de scanare ----
        # OS-ul va da intaietate UI-ului si altor aplicatii. Calculatorul
        # ramane responsiv chiar si in timpul scanarii unei colectii mari.
        self.scanner_activ.start(QThread.Priority.LowPriority)

    def _dupa_scanare_finalizata(self):
        self.incarca_index_faiss()
        self.actualizeaza_smart_albums()
        self.afiseaza_toata_libraria()

    # ----------------------------------------------------------
    # SURSE (foldere importate)
    # ----------------------------------------------------------

    def adauga_sursa_noua(self):
        cale = QFileDialog.getExistingDirectory(self.window, "Selecteaza folderul")
        if not cale:
            return
        cale = cale.replace("\\", "/")
        self.db.adauga_sursa(cale)
        self.incarca_sursele_vizual()

        check = self.window.findChild(QtWidgets.QCheckBox, "checkRecursive")
        recursiv = check.isChecked() if check else False
        self.porneste_scanare_folder(cale, recursiv)

    def sterge_sursa_selectata(self):
        lista = self.window.findChild(QtWidgets.QListWidget, "sourceListWidget")
        item = lista.currentItem()
        if not item:
            return
        if QMessageBox.question(self.window, "Stergere", f"Elimini {item.text()}?") == QMessageBox.Yes:
            self.db.sterge_sursa_si_imagini(item.text())
            self.incarca_sursele_vizual()
            self.afiseaza_toata_libraria()
            self.actualizeaza_smart_albums()

    def incarca_sursele_vizual(self):
        lista = self.window.findChild(QtWidgets.QListWidget, "sourceListWidget")
        if not lista:
            return
        lista.clear()
        for s in self.db.obtine_surse():
            item = QtWidgets.QListWidgetItem(s)
            item.setIcon(self.window.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
            lista.addItem(item)

    def cand_apas_pe_sursa(self, item: QtWidgets.QListWidgetItem):
        self.vizualizare_activa = "folder"
        toate = self.db.obtine_toate_caile_existente()
        filtrate = [p for p in toate if p.startswith(item.text())]
        self.populeaza_galeria_cu_cai(filtrate)

    # ----------------------------------------------------------
    # ALBUME INTELIGENTE (TreeWidget)
    # ----------------------------------------------------------

    def actualizeaza_smart_albums(self):
        """Reconstruieste QTreeWidget cu ierarhia predefinita + albumele custom din DB."""
        tree = self.window.findChild(QtWidgets.QTreeWidget, "smartTreeWidget")
        if not tree:
            return

        tree.clear()
        tree.setHeaderLabel("Organizare Inteligenta")
        tree.setIndentation(20)

        # ---- OPTIMIZARE: un singur query pentru toate numaratorile ----
        # Inainte: 16 query-uri SQL (cate unul per subcategorie)
        # Acum:    1 query SQL cu GROUP BY
        numaratori = self.db.numara_per_categorii_toate()

        # A. Categorii predefinite (clasificate de CLIP la scanare)
        for domeniu, categorii in STRUCTURA_ALBUME.items():
            domeniu_item = QtWidgets.QTreeWidgetItem([domeniu])
            font = domeniu_item.font(0)
            font.setBold(True)
            domeniu_item.setFont(0, font)
            tree.addTopLevelItem(domeniu_item)

            for cat, subcategorii in categorii.items():
                cat_item = QtWidgets.QTreeWidgetItem([cat])
                domeniu_item.addChild(cat_item)

                for sub in subcategorii:
                    nr = numaratori.get(sub, 0)
                    sub_item = QtWidgets.QTreeWidgetItem([f"{sub} ({nr})"])
                    sub_item.setData(0, Qt.UserRole, sub)
                    cat_item.addChild(sub_item)

        # B. Albume custom (salvate in DB, persistente intre sesiuni)
        albume_custom = self.db.obtine_albume_custom()
        if albume_custom:
            separator = QtWidgets.QTreeWidgetItem(["── Smart Albums ──"])
            separator.setFlags(Qt.NoItemFlags)
            tree.addTopLevelItem(separator)

            for nume in albume_custom:
                item = QtWidgets.QTreeWidgetItem([f"✦ {nume}"])
                item.setData(0, Qt.UserRole, f"SEARCH:{nume}")
                font = item.font(0)
                font.setItalic(True)
                item.setFont(0, font)
                tree.addTopLevelItem(item)

        tree.expandAll()

    def cand_apas_pe_smart_album_tree(self, item: QtWidgets.QTreeWidgetItem, col: int):
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        self._reseteaza_filtru()

        if data.startswith("SEARCH:"):
            termen = data.replace("SEARCH:", "")
            self._afiseaza_rezultate_cautare(termen)
        else:
            cai = self.db.obtine_cai_dupa_categorie(data)
            self.populeaza_galeria_cu_cai(cai)

    def creeaza_album_inteligent(self):
        """
        Deschide un dialog pentru numele albumului, il salveaza in DB
        si il adauga in TreeWidget. Albumul persista intre sesiuni.
        """
        nume, ok = QtWidgets.QInputDialog.getText(
            self.window,
            "Album Nou",
            "Descrie ce cauti (ex: pisici albe, apus de soare, munte iarna):"
        )
        if not (ok and nume.strip()):
            return

        nume = nume.strip()
        self.db.salveaza_album_custom(nume)
        self.actualizeaza_smart_albums()
        self._afiseaza_rezultate_cautare(nume)
        print(f"[UI] Album Smart creat si salvat: '{nume}'")

    # ----------------------------------------------------------
    # GALERIE
    # ----------------------------------------------------------

    def afiseaza_toata_libraria(self):
        self.vizualizare_activa = "librarie"
        self._reseteaza_filtru()
        self.populeaza_galeria_cu_cai(self.db.obtine_toate_caile_existente())

    def populeaza_galeria_cu_cai(self, cai: list[str]):
        """Incarca lista de imagini in QListView folosind thumbnail-urile din cache."""
        self.model_galerie.clear()

        cai_unice = list(dict.fromkeys(
            os.path.normpath(c).replace("\\", "/") for c in cai
        ))

        # ---- OPTIMIZARE: un singur query in loc de N ----
        # Inainte: cate un query SQL per imagine (cauta_dupa_cale) → cu 5000
        #          de imagini, 5000 de query-uri si UI inghetat 2-5 secunde.
        # Acum:    un singur dict cu toate {cale: cale_cache}.
        mapa_cache = self.db.obtine_cai_si_cache()

        for cale in cai_unice:
            if not os.path.exists(cale):
                continue
            item = QStandardItem(os.path.basename(cale))
            cache = mapa_cache.get(cale)
            icon_path = cache if (cache and os.path.exists(cache)) else cale
            item.setData(QIcon(icon_path), Qt.ItemDataRole.DecorationRole)
            item.setData(cale, Qt.ItemDataRole.UserRole)
            self.model_galerie.appendRow(item)

    def cand_selectez_o_imagine(self, index: QtCore.QModelIndex):
        """Citeste datele din DB si actualizeaza panoul de detalii din dreapta."""
        # 1. Obtinem calea reala a fisierului trecand de filtrul proxy
        index_sursa = self.proxy_model.mapToSource(index)
        cale = index_sursa.data(Qt.ItemDataRole.UserRole)

        if not cale:
            return

        # 2. Cautam informatiile in baza de date
        date_db = self.db.cauta_dupa_cale(cale)
        preview_label = self.window.findChild(QtWidgets.QLabel, "previewLabel")

        if date_db:
            # 3. Pregatim dictionarul de info (folosind accesul prin nume/cheie)
            info = {
                "nume":      date_db['nume'],
                "rezolutie": date_db['rezolutie'],
                "mb":        date_db['mb'],
                "marca":     date_db['marca'],
                "model":     date_db['model'],
                "data":      date_db['data_poza'],
                "gps":       date_db['gps'],
                "oras":      date_db['oras'],
                "tara":      date_db['tara'],
            }

            # 4. Gestionam vizualizarea (Thumbnail din cache sau imagine originala)
            cale_cache = date_db['cale_cache']
            pixmap = (
                QtGui.QPixmap(cale_cache)
                if (cale_cache and os.path.exists(cale_cache))
                else QtGui.QPixmap(cale)
            )

            # 5. Actualizam interfata
            self._actualizeaza_panou_dreapta(info, pixmap)

        else:
            # 6. Fallback: Daca imaginea nu este inca in DB, o procesam live
            if self.procesor_activ and self.procesor_activ.isRunning():
                self.procesor_activ.terminate()
                self.procesor_activ.wait()

            self.procesor_activ = ProcesorImagine(cale, preview_label.size())
            self.procesor_activ.gata_procesarea.connect(self._actualizeaza_panou_dreapta)
            self.procesor_activ.start()

    def deschide_poza_nativ(self, index: QtCore.QModelIndex):
        cale = self.proxy_model.mapToSource(index).data(Qt.ItemDataRole.UserRole)
        if cale and os.path.exists(cale):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(cale))

    def sterge_imaginea_selectata(self):
        idx = self.view_galerie.currentIndex()
        if not idx.isValid():
            return
        idx_s = self.proxy_model.mapToSource(idx)
        cale = idx_s.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self.window, "Stergere", "Stergi imaginea din baza de date?") == QMessageBox.Yes:
            self.db.sterge_imagine_dupa_cale(cale)
            self.model_galerie.removeRow(idx_s.row())
            self.incarca_index_faiss()

    def arata_meniu_poza(self, poz: QtCore.QPoint):
        idx = self.view_galerie.indexAt(poz)
        if not idx.isValid():
            return
        m = QtWidgets.QMenu()
        act = m.addAction("Gaseste poze similare (AI)")
        if m.exec(self.view_galerie.mapToGlobal(poz)) == act:
            self._executa_cautare_similara(idx)

    def _executa_cautare_similara(self, index: QtCore.QModelIndex):
        """Cautare imagine-la-imagine: gaseste pozele cel mai apropiate vectorial."""
        idx_s = self.proxy_model.mapToSource(index)
        cale = idx_s.data(Qt.ItemDataRole.UserRole)
        date = self.db.cauta_dupa_cale(cale)
        if not date or not date[COL_VECTOR_AI]:
            return

        v_q = np.array(pickle.loads(date[COL_VECTOR_AI])).astype("float32").reshape(1, -1)
        faiss.normalize_L2(v_q)
        dist, idxs = self.index_faiss.search(v_q, min(10, self.index_faiss.ntotal))

        cai_similare = [
            self.mapare_cai[x]
            for x, scor in zip(idxs[0], dist[0])
            if x != -1 and scor > PRAG_SIMILITUDINE_VIZUALA
        ]
        self._filtreaza_galerie_dupa_cai(cai_similare)

    # ----------------------------------------------------------
    # CAUTARE
    # ----------------------------------------------------------

    def aplic_filtrare_simpla(self, text: str):
        """Filtru instant dupa numele fisierului (fara AI)."""
        self.proxy_model.setFilterFixedString(text)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

    def execut_cautare_ai(self):
        """Apasare Enter in search bar → cautare semantica CLIP + FAISS."""
        text = self.search_bar.text().strip()
        if not text:
            self._reseteaza_filtru()
            self.afiseaza_toata_libraria()
            return
        self._afiseaza_rezultate_cautare(text)

    def _afiseaza_rezultate_cautare(self, text: str):
        """Motor comun pentru cautare semantica (folosit de search bar si albume custom)."""
        cai = self.cauta_semantic(text)
        if cai:
            self._filtreaza_galerie_dupa_cai(cai)
            self.window.statusBar().showMessage(f"'{text}' → {len(cai)} rezultate")
        else:
            self.proxy_model.setFilterFixedString("___NIMIC_GASIT___")
            self.window.statusBar().showMessage(f"Niciun rezultat pentru '{text}'")

    def _filtreaza_galerie_dupa_cai(self, cai: list[str]):
        """Seteaza un regex pe proxy_model pentru a afisa doar caile specificate."""
        if not cai:
            return
        nume_escaped = [
            QtCore.QRegularExpression.escape(os.path.basename(c)) for c in cai
        ]
        pattern = "^(" + "|".join(nume_escaped) + ")$"
        opts = QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption
        self.proxy_model.setFilterRegularExpression(
            QtCore.QRegularExpression(pattern, opts)
        )

    def _reseteaza_filtru(self):
        """Curata absolut orice filtru activ pentru a lasa galeria libera."""
        self.proxy_model.setFilterFixedString("")
        self.proxy_model.setFilterRegularExpression("")
        if hasattr(self, 'search_bar'):
            self.search_bar.clear()

    # ----------------------------------------------------------
    # ORGANIZARE FIZICA PE DISC
    # ----------------------------------------------------------

    def executa_organizarea_fizica(self):
        """Organizare: Domeniu -> Categorie -> Subcategorie -> An-Luna -> Tara -> Oras."""
        destinatie = QFileDialog.getExistingDirectory(self.window, "Selecteaza destinatia de export")
        if not destinatie:
            return

        date_poze = self.db.obtine_toate_pentru_organizare()
        if not date_poze:
            return

        progress = QProgressDialog("Se organizeaza colectia...", "Anuleaza", 0, len(date_poze), self.window)
        progress.show()

        succes = 0
        for i, rand in enumerate(date_poze):
            if progress.wasCanceled():
                break

            # Despachetam cele 8 coloane returnate de DB
            cale_orig, subcat_ai, data_raw, gps_raw, lat, lon, oras, tara = rand

            # 1. Identificam ierarhia din STRUCTURA_ALBUME
            domeniu, categorie_mare = "Alte_Categorii", "Diverse"
            subcat_finala = subcat_ai if subcat_ai else "Nesortate"

            gasit = False
            for dom, categorii in STRUCTURA_ALBUME.items():
                for cat_m, lista_subs in categorii.items():
                    if subcat_finala in lista_subs:
                        domeniu = dom
                        categorie_mare = cat_m
                        gasit = True
                        break
                if gasit:
                    break

            # 2. Data → An-Luna (ex: "2024-06")
            if data_raw and len(data_raw) >= 7:
                an   = data_raw[:4]
                luna = data_raw[5:7]
                data_folder = f"{an}-{luna}"
            else:
                data_folder = "Data_Necunoscuta"

            # 3. Locatie din geocodare: Tara/Oras
            if tara and oras and tara not in ("Necunoscut", ""):
                locatie_folder = os.path.join(
                    self._curata_pentru_folder(tara),
                    self._curata_pentru_folder(oras)
                )
            else:
                locatie_folder = "Fara_Locatie"

            # 4. Construim calea finala
            cale_relativa = os.path.join(
                domeniu, categorie_mare, subcat_finala, data_folder, locatie_folder
            )
            folder_final = os.path.join(destinatie, cale_relativa).replace("\\", "/")

            try:
                os.makedirs(folder_final, exist_ok=True)
                if os.path.exists(cale_orig):
                    shutil.copy2(cale_orig, os.path.join(folder_final, os.path.basename(cale_orig)))
                    succes += 1
            except Exception as e:
                print(f"[EROARE organizare] {e}")

            progress.setValue(i + 1)

        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(destinatie))
        QMessageBox.information(self.window, "Succes", f"Organizare gata! {succes} poze copiate.")

    @staticmethod
    def _curata_pentru_folder(text: str | None) -> str:
        """Elimina caracterele interzise pe Windows dintr-un string destinat numelui de folder."""
        if not text:
            return "Necunoscut"
        caractere_interzise = [":", "|", "/", "\\", "<", ">", "*", "?", '"', "."]
        rezultat = text
        for c in caractere_interzise:
            rezultat = rezultat.replace(c, "-")
        return rezultat.strip()

    @staticmethod
    def _curata_gps_pentru_folder(gps_raw: str | None) -> str:
        """Pastrat pentru compatibilitate. Foloseste _curata_pentru_folder in locul sau."""
        if not gps_raw or gps_raw in ("", "Fara GPS"):
            return "Fara_Locatie"
        return MainWindow._curata_pentru_folder(gps_raw)

    # ----------------------------------------------------------
    # HELPERS UI
    # ----------------------------------------------------------

    def _actualizeaza_panou_dreapta(self, date: dict, pixmap: QtGui.QPixmap):
        """Afiseaza thumbnailul si metadatele in panoul din dreapta."""
        preview_label = self.window.findChild(QtWidgets.QLabel, "previewLabel")
        info_label    = self.window.findChild(QtWidgets.QLabel, "infoLabel")
        if not preview_label or not info_label:
            return

        if pixmap and not pixmap.isNull():
            preview_label.setPixmap(
                pixmap.scaled(preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        linii = [
            f"<b>Fisier:</b> {date.get('nume', '---')}",
            f"<b>Marime:</b> {date.get('mb', '0')} MB",
            f"<b>Rezolutie:</b> {date.get('rezolutie', '---')}",
        ]

        marca = date.get("marca")
        model = date.get("model")
        if marca and marca != "Necunoscut":
            linii.append(f"<b>Echipament:</b> {marca} {model or ''}")

        data_p = date.get("data")
        if data_p and data_p not in ("Data Necunoscuta", "---", None):
            linii.append(f"<b>Data:</b> {data_p}")

        # Locatie: afisam Oras, Tara daca exista, altfel GPS brut
        oras = date.get("oras")
        tara = date.get("tara")
        gps_p = date.get("gps")
        if oras and tara and tara not in ("Necunoscut", ""):
            linii.append(f"<b>Locatie:</b> {oras}, {tara}")
        elif gps_p and gps_p not in ("Fara GPS", ""):
            linii.append(f"<b>GPS:</b> {gps_p}")

        info_label.setText("<br>".join(linii))
        info_label.setWordWrap(True)

    def _updateaza_status_progres(self, curent: int, total: int):
        if self.window.statusBar():
            self.window.statusBar().showMessage(f"Scanare AI: {curent}/{total}")

    def _actualizeaza_iconita_live(self, index_0based: int):
        """Actualizeaza thumbnail-ul in galerie imediat dupa ce e procesat."""
        idx = self.model_galerie.index(index_0based - 1, 0)
        if idx.isValid():
            cale = idx.data(Qt.ItemDataRole.UserRole)
            d = self.db.cauta_dupa_cale(cale)
            if d and d[COL_CALE_CACHE]:
                self.model_galerie.setData(idx, QIcon(d[COL_CALE_CACHE]), Qt.ItemDataRole.DecorationRole)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    controller = MainWindow()
    sys.exit(app.exec())
