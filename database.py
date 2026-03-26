import sqlite3
import pickle
import numpy as np

class ManagerBazaDate:
    def __init__(self, cale_db):
        self.nume_db = cale_db 
        self.creeaza_tabel()

    def _conectare(self):
        """Deschide conexiunea cu fisierul DB folosind calea salvata."""
        return sqlite3.connect(self.nume_db)

    def creeaza_tabel(self):
        """Creeaza tabelele pentru imagini si pentru sursele de import."""
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
        
        # Tabelul nou pentru folderele sursa (Libraria ta)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS surse (
                cale TEXT PRIMARY KEY
            )
        ''')
        
        conn.commit()
        conn.close()

    # --- FUNCTII PENTRU MANAGEMENTUL SURSELOR (LIBRARIE) ---

    def adauga_sursa(self, cale_folder):
        """Salveaza un folder nou in lista de surse a aplicatiei."""
        conn = self._conectare()
        try:
            conn.execute("INSERT OR IGNORE INTO surse (cale) VALUES (?)", (cale_folder,))
            conn.commit()
        finally:
            conn.close()

    def obtine_surse(self):
        """Returneaza toate folderele pe care utilizatorul le-a adaugat in librarie."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM surse")
        rezultate = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rezultate

    def sterge_sursa(self, cale_folder):
        """Elimina un folder din lista de surse (nu sterge pozele de pe disc)."""
        conn = self._conectare()
        conn.execute("DELETE FROM surse WHERE cale = ?", (cale_folder,))
        conn.commit()
        conn.close()

    def obtine_toate_caile_existente(self):
        """Returneaza absolut toate pozele din DB pentru afisarea 'All Photos'."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM imagini")
        rezultate = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rezultate

    # --- FUNCTII DE SALVARE SI CAUTARE (EXISTENTE) ---

    def salveaza_sau_actualizeaza(self, d):
        conn = self._conectare()
        cursor = conn.cursor()
        vector_binar = pickle.dumps(d.get('vector_ai')) if d.get('vector_ai') is not None else None

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO imagini 
                (cale, nume, format, rezolutie, mb, marca, model, data_poza, gps, cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                d.get('cale'), d.get('nume'), d.get('format'), d.get('rezolutie'), 
                d.get('mb'), d.get('marca'), d.get('model'), d.get('data'), 
                d.get('gps'), d.get('cale_cache'), d.get('categorie'), vector_binar
            ))
            conn.commit()
        except Exception as e:
            print(f"Eroare SQLite la salvare: {e}")
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
                date_ai.append((cale, pickle.loads(v_binar)))
            except: continue
        return date_ai

    def numara_per_categorie(self, nume_categorie):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM imagini WHERE categorie = ?", (nume_categorie,))
        rezultat = cursor.fetchone()
        conn.close()
        return rezultat[0] if rezultat else 0

    def obtine_cai_dupa_categorie(self, nume_categorie):
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM imagini WHERE categorie = ?", (nume_categorie,))
        rezultate = cursor.fetchall()
        conn.close()
        return [r[0] for r in rezultate]
    
    def sterge_sursa_si_imagini(self, cale_folder):
        """Sterge folderul din surse si toate imaginile care incep cu acea cale."""
        conn = self._conectare()
        try:
            cursor = conn.cursor()
            # 1. Stergem folderul din lista de surse
            cursor.execute("DELETE FROM surse WHERE cale = ?", (cale_folder,))
            # 2. Stergem toate imaginile care apartin de acel folder
            # Folosim LIKE 'cale/%' pentru a prinde tot ce e inauntru
            cursor.execute("DELETE FROM imagini WHERE cale LIKE ?", (f"{cale_folder}%",))
            conn.commit()
        finally:
            conn.close()
    
    def obtine_toate_pentru_organizare(self):
        """Returneaza o lista de tuple (cale_originala, categorie) pentru export."""
        conn = self._conectare()
        cursor = conn.cursor()
        # Luam calea si categoria pentru toate imaginile scanate
        cursor.execute("SELECT cale, categorie FROM imagini")
        rezultate = cursor.fetchall()
        conn.close()
        return rezultate
    
    def sterge_imagine_dupa_cale(self, cale):
        """Sterge o singura imagine din baza de date folosind calea ei."""
        conn = self._conectare()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM imagini WHERE cale = ?", (cale,))
            conn.commit()
        finally:
            conn.close()
    