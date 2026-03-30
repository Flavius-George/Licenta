import sqlite3
import pickle
import numpy as np

class ManagerBazaDate:
    def __init__(self, cale_db):
        """Initializam baza de date si cream tabelele."""
        self.nume_db = cale_db 
        self.creeaza_tabel()

    def _conectare(self):
        """Deschide conexiunea cu fisierul DB."""
        return sqlite3.connect(self.nume_db)

    def creeaza_tabel(self):
        """Creeaza structura tabelelor si indexurile pentru viteza."""
        conn = self._conectare()
        cursor = conn.cursor()
        
        # Tabelul principal de imagini
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS imagini (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cale TEXT UNIQUE,
                nume TEXT,
                format TEXT,
                rezolutie TEXT,
                mb REAL,
                marca TEXT,
                model TEXT,
                data_poza TEXT,
                gps TEXT,
                cale_cache TEXT,
                categorie TEXT,
                vector_ai BLOB
            )
        ''')
        
        # Tabelul pentru folderele sursa
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS surse (
                cale TEXT PRIMARY KEY
            )
        ''')

        # Indexuri pentru cautare instanta
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categorie ON imagini(categorie)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cale ON imagini(cale)")
        
        conn.commit()
        conn.close()

    def adauga_sursa(self, cale_folder):
        conn = self._conectare()
        try:
            conn.execute("INSERT OR IGNORE INTO surse (cale) VALUES (?)", (cale_folder,))
            conn.commit()
        finally:
            conn.close()

    def obtine_surse(self):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM surse")
        rezultate = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rezultate

    def sterge_sursa_si_imagini(self, cale_folder):
        conn = self._conectare()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM surse WHERE cale = ?", (cale_folder,))
            cursor.execute("DELETE FROM imagini WHERE cale LIKE ?", (f"{cale_folder}%",))
            conn.commit()
        finally:
            conn.close()

    def salveaza_sau_actualizeaza(self, d):
        conn = self._conectare()
        v_binar = pickle.dumps(d.get('vector_ai')) if d.get('vector_ai') is not None else None
        try:
            conn.execute('''
                INSERT OR REPLACE INTO imagini 
                (cale, nume, format, rezolutie, mb, marca, model, data_poza, gps, cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                d.get('cale'), d.get('nume'), d.get('format'), d.get('rezolutie'), 
                d.get('mb'), d.get('marca'), d.get('model'), d.get('data'), 
                d.get('gps'), d.get('cale_cache'), d.get('categorie'), v_binar
            ))
            conn.commit()
        except Exception as e:
            print(f"Eroare SQLite: {e}")
        finally:
            conn.close()

    def cauta_dupa_cale(self, cale_fisier):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM imagini WHERE cale = ?", (cale_fisier,))
        date = cursor.fetchone()
        conn.close()
        return date

    def obtine_toti_vectorii(self):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale, vector_ai FROM imagini WHERE vector_ai IS NOT NULL")
        rezultate = cursor.fetchall()
        conn.close()
        date_ai = []
        for cale, v_binar in rezultate:
            try:
                date_ai.append((cale, pickle.loads(v_binar).astype('float32')))
            except: continue
        return date_ai

    def numara_per_categorie(self, nume_categorie):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM imagini WHERE categorie = ?", (nume_categorie,))
        rez = cursor.fetchone()
        conn.close()
        return rez[0] if rez else 0

    def obtine_cai_dupa_categorie(self, nume_categorie):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM imagini WHERE categorie = ?", (nume_categorie,))
        rezultate = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rezultate

    def obtine_toate_caile_existente(self):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM imagini")
        rezultate = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rezultate

    def sterge_imagine_dupa_cale(self, cale):
        conn = self._conectare()
        try:
            conn.execute("DELETE FROM imagini WHERE cale = ?", (cale,))
            conn.commit()
        finally:
            conn.close()

        # --- FUNCTIA CARE LIPSEA SI PROVOCA EROAREA ---
    def obtine_toate_pentru_organizare(self):
        """Returneaza Cale, Categorie, Data si GPS pentru organizare complexa."""
        conn = self._conectare()
        cursor = conn.cursor()
        # Luam coloanele 1 (cale), 11 (categorie), 8 (data_poza) si 9 (gps)
        cursor.execute("SELECT cale, categorie, data_poza, gps FROM imagini")
        rezultate = cursor.fetchall()
        conn.close()
        return rezultate

    def reset_total(self):
        conn = self._conectare()
        conn.execute("DELETE FROM imagini")
        conn.execute("DELETE FROM surse")
        conn.commit()
        conn.close()