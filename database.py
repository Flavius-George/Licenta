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
    """

    def __init__(self, cale_db: str):
        self.cale_db = cale_db
        self._creeaza_tabele()

    # ----------------------------------------------------------
    # CONEXIUNE
    # ----------------------------------------------------------

    def _conectare(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cale_db)

    # ----------------------------------------------------------
    # INITIALIZARE SCHEMA
    # ----------------------------------------------------------

    def _creeaza_tabele(self):
        """Creeaza tabelele si indexurile la prima rulare."""
        with self._conectare() as conn:
            conn.executescript("""
                -- Tabelul principal: o linie per imagine
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
                    cale_cache  TEXT,
                    categorie   TEXT,
                    vector_ai   BLOB
                );

                -- Folderele adaugate de utilizator ca surse
                CREATE TABLE IF NOT EXISTS surse (
                    cale TEXT PRIMARY KEY
                );

                -- Albumele Smart create manual de utilizator
                -- Persistente intre sesiuni!
                CREATE TABLE IF NOT EXISTS albume_custom (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    nume      TEXT UNIQUE NOT NULL,
                    creat_la  TEXT DEFAULT (datetime('now'))
                );

                -- Indexuri pentru cautare rapida
                CREATE INDEX IF NOT EXISTS idx_categorie ON imagini(categorie);
                CREATE INDEX IF NOT EXISTS idx_cale      ON imagini(cale);
            """)

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
                     cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                d.get("cale"),    d.get("nume"),      d.get("format"),
                d.get("rezolutie"), d.get("mb"),      d.get("marca"),
                d.get("model"),   d.get("data"),      d.get("gps"),
                d.get("cale_cache"), d.get("categorie"), v_binar,
            ))

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

    def obtine_toate_pentru_organizare(self) -> list[tuple]:
        """Returneaza (cale, categorie, data_poza, gps) pentru organizare pe disc."""
        with self._conectare() as conn:
            return conn.execute(
                "SELECT cale, categorie, data_poza, gps FROM imagini"
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
