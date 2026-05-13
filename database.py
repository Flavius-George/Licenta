import sqlite3
import pickle
import numpy as np


class ManagerBazaDate:
    """
    Layer de acces la date (DAL) pentru aplicatia GalerieLicentaAI.

    Tabele gestionate:
    - imagini       : metadate + embedding CLIP pentru fiecare imagine
    - surse         : folderele importate de utilizator
    - albume_custom : albumele inteligente create manual de utilizator

    Schema imagini (in ordine, pentru referinta indecshilor COL_*):
        0  id
        1  cale
        2  nume
        3  format
        4  rezolutie
        5  mb
        6  marca
        7  model
        8  data_poza
        9  gps
        10 lat
        11 lon
        12 oras
        13 tara
        14 tara_cod
        15 cale_cache
        16 categorie
        17 vector_ai

    OPTIMIZARI PERFORMANTA:
    - PRAGMA journal_mode=WAL          → scrieri concurente, fara blocare cititorilor
    - PRAGMA synchronous=NORMAL        → reduce fsync-urile (de 5-10x mai rapid la insert)
    - PRAGMA temp_store=MEMORY         → tabele temporare in RAM
    - PRAGMA cache_size=-20000         → 20MB cache de pagini
    - Metode bulk pentru a evita N+1 queries (numara_per_categorii_toate, obtine_cai_si_cache)
    """

    def __init__(self, cale_db: str):
        self.cale_db = cale_db
        self._creeaza_tabele()

    # ----------------------------------------------------------
    # CONEXIUNE
    # ----------------------------------------------------------

    def _conectare(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.cale_db)
        conn.row_factory = sqlite3.Row  # Transformă rândurile în obiecte tip dicționar
        # ---- PRAGMAs de performanta ----
        # WAL: scrieri concurente; persistent pe disc (setat o data, ramane setat)
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: nu mai face fsync dupa fiecare commit (de 5-10x mai rapid la insert)
        conn.execute("PRAGMA synchronous=NORMAL")
        # Tabelele/indecsii temporari raman in RAM
        conn.execute("PRAGMA temp_store=MEMORY")
        # 20MB de cache pe pagini
        conn.execute("PRAGMA cache_size=-20000")
        return conn

    # ----------------------------------------------------------
    # INITIALIZARE SCHEMA
    # ----------------------------------------------------------

    def _creeaza_tabele(self):
        """Creeaza tabelele si indexurile la prima rulare."""
        with self._conectare() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS imagini (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    cale        TEXT    UNIQUE,
                    nume        TEXT,
                    format      TEXT,
                    rezolutie   TEXT,
                    mb          REAL,
                    marca       TEXT,
                    model       TEXT,
                    data_poza   TEXT,
                    gps         TEXT,
                    lat         REAL,
                    lon         REAL,
                    oras        TEXT,
                    tara        TEXT,
                    tara_cod    TEXT,
                    cale_cache  TEXT,
                    categorie   TEXT,
                    vector_ai   BLOB
                );

                CREATE TABLE IF NOT EXISTS surse (
                    cale TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS albume_custom (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    nume      TEXT UNIQUE NOT NULL,
                    creat_la  TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_categorie ON imagini(categorie);
                CREATE INDEX IF NOT EXISTS idx_cale      ON imagini(cale);
            """)

            # Migrare pentru baze de date existente (vechi fara coloanele noi)
            for coloana, tip in [
                ("lat",      "REAL"),
                ("lon",      "REAL"),
                ("oras",     "TEXT"),
                ("tara",     "TEXT"),
                ("tara_cod", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE imagini ADD COLUMN {coloana} {tip}")
                    conn.commit()
                except Exception:
                    pass  # coloana deja exista

    # ----------------------------------------------------------
    # IMAGINI
    # ----------------------------------------------------------

    def salveaza_sau_actualizeaza(self, d: dict):
        """Insereaza sau actualizeaza (UPSERT) o imagine in baza de date."""
        v_binar = pickle.dumps(d["vector_ai"]) if d.get("vector_ai") is not None else None
        with self._conectare() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO imagini
                    (cale, nume, format, rezolutie, mb,
                     marca, model, data_poza, gps,
                     lat, lon, oras, tara, tara_cod,
                     cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d.get("cale"),
                d.get("nume"),
                d.get("format"),
                d.get("rezolutie"),
                d.get("mb"),
                d.get("marca"),
                d.get("model"),
                d.get("data"),
                d.get("gps"),
                d.get("lat"),
                d.get("lon"),
                d.get("oras"),
                d.get("tara"),
                d.get("tara_cod"),
                d.get("cale_cache"),
                d.get("categorie"),
                v_binar,
            ))

    def salveaza_batch(self, lista_dict: list[dict]):
        """
        Insereaza un BATCH de imagini intr-o singura tranzactie.
        De 10-50x mai rapid decat apelurile individuale, mai ales in
        timpul scanarii. Folosit de ScannerWorker.
        """
        if not lista_dict:
            return
        randuri = []
        for d in lista_dict:
            v_binar = pickle.dumps(d["vector_ai"]) if d.get("vector_ai") is not None else None
            randuri.append((
                d.get("cale"), d.get("nume"), d.get("format"), d.get("rezolutie"),
                d.get("mb"), d.get("marca"), d.get("model"), d.get("data"),
                d.get("gps"), d.get("lat"), d.get("lon"), d.get("oras"),
                d.get("tara"), d.get("tara_cod"), d.get("cale_cache"),
                d.get("categorie"), v_binar,
            ))
        with self._conectare() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO imagini
                    (cale, nume, format, rezolutie, mb,
                     marca, model, data_poza, gps,
                     lat, lon, oras, tara, tara_cod,
                     cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, randuri)

    def cauta_dupa_cale(self, cale: str) -> tuple | None:
        """Returneaza tupla completa a imaginii sau None daca nu exista."""
        with self._conectare() as conn:
            return conn.execute(
                "SELECT * FROM imagini WHERE cale = ?", (cale,)
            ).fetchone()

    def obtine_toti_vectorii(self) -> list[tuple[str, np.ndarray]]:
        """
        Returneaza perechile (cale, vector_numpy) pentru toate imaginile
        care au un embedding CLIP stocat.
        """
        with self._conectare() as conn:
            rows = conn.execute(
                "SELECT cale, vector_ai FROM imagini WHERE vector_ai IS NOT NULL"
            ).fetchall()

        rezultate = []
        for cale, v_binar in rows:
            try:
                rezultate.append((cale, pickle.loads(v_binar).astype("float32")))
            except Exception:
                continue
        return rezultate

    def numara_per_categorie(self, categorie: str) -> int:
        with self._conectare() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM imagini WHERE categorie = ?", (categorie,)
            ).fetchone()
        return row[0] if row else 0

    def numara_per_categorii_toate(self) -> dict[str, int]:
        """
        OPTIMIZARE: returneaza intr-un singur query numarul de imagini din
        FIECARE categorie (in loc de un query separat per subcategorie, care
        producea 16 round-trip-uri la fiecare refresh al smart albums).
        """
        with self._conectare() as conn:
            rows = conn.execute(
                "SELECT categorie, COUNT(*) AS cnt FROM imagini "
                "WHERE categorie IS NOT NULL GROUP BY categorie"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def obtine_cai_dupa_categorie(self, categorie: str) -> list[str]:
        with self._conectare() as conn:
            rows = conn.execute(
                "SELECT cale FROM imagini WHERE categorie = ?", (categorie,)
            ).fetchall()
        return [r[0] for r in rows]

    def obtine_toate_caile_existente(self) -> list[str]:
        with self._conectare() as conn:
            rows = conn.execute("SELECT cale FROM imagini").fetchall()
        return [r[0] for r in rows]

    def obtine_cai_si_cache(self) -> dict[str, str | None]:
        """
        OPTIMIZARE: returneaza dict {cale: cale_cache} pentru TOATE imaginile
        intr-un singur query. Inlocuieste apelurile per-imagine la
        cauta_dupa_cale() din populeaza_galeria_cu_cai().
        """
        with self._conectare() as conn:
            rows = conn.execute("SELECT cale, cale_cache FROM imagini").fetchall()
        return {r[0]: r[1] for r in rows}

    def obtine_set_complet_procesate(self) -> set[str]:
        """
        Returneaza setul de cai care au DEJA vector AI si thumbnail in DB.
        Folosit de scanner pentru a sari peste imaginile deja procesate
        FARA a mai incarca modelul CLIP daca nu e nimic de facut.
        """
        with self._conectare() as conn:
            rows = conn.execute(
                "SELECT cale FROM imagini "
                "WHERE vector_ai IS NOT NULL "
                "AND cale_cache IS NOT NULL "
                "AND cale_cache != ''"
            ).fetchall()
        return {r[0] for r in rows}

    def obtine_toate_pentru_organizare(self) -> list[tuple]:
        """Returneaza (cale, categorie, data_poza, gps, lat, lon, oras, tara) pentru organizare pe disc."""
        with self._conectare() as conn:
            return conn.execute(
                "SELECT cale, categorie, data_poza, gps, lat, lon, oras, tara FROM imagini"
            ).fetchall()

    def sterge_imagine_dupa_cale(self, cale: str):
        with self._conectare() as conn:
            conn.execute("DELETE FROM imagini WHERE cale = ?", (cale,))

    # ----------------------------------------------------------
    # SURSE
    # ----------------------------------------------------------

    def adauga_sursa(self, cale_folder: str):
        with self._conectare() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO surse (cale) VALUES (?)", (cale_folder,)
            )

    def obtine_surse(self) -> list[str]:
        with self._conectare() as conn:
            rows = conn.execute("SELECT cale FROM surse").fetchall()
        return [r[0] for r in rows]

    def sterge_sursa_si_imagini(self, cale_folder: str):
        with self._conectare() as conn:
            conn.execute("DELETE FROM surse  WHERE cale = ?", (cale_folder,))
            conn.execute("DELETE FROM imagini WHERE cale LIKE ?", (f"{cale_folder}%",))

    # ----------------------------------------------------------
    # ALBUME CUSTOM (persistente intre sesiuni)
    # ----------------------------------------------------------

    def salveaza_album_custom(self, nume: str):
        """Salveaza un album inteligent creat de utilizator."""
        with self._conectare() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO albume_custom (nume) VALUES (?)", (nume,)
            )

    def obtine_albume_custom(self) -> list[str]:
        """Returneaza toate albumele custom ordonate dupa data crearii."""
        with self._conectare() as conn:
            rows = conn.execute(
                "SELECT nume FROM albume_custom ORDER BY creat_la ASC"
            ).fetchall()
        return [r[0] for r in rows]

    def sterge_album_custom(self, nume: str):
        with self._conectare() as conn:
            conn.execute(
                "DELETE FROM albume_custom WHERE nume = ?", (nume,)
            )

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------

    def reset_total(self):
        """Sterge toate datele (folosit pentru debug / reinstalare)."""
        with self._conectare() as conn:
            conn.executescript("""
                DELETE FROM imagini;
                DELETE FROM surse;
                DELETE FROM albume_custom;
            """)
