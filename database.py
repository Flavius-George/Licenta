import sqlite3
import pickle
import numpy as np

class ManagerBazaDate:
    def __init__(self, nume_db="galerie_licenta.db"):
        self.nume_db = nume_db
        self.creeaza_tabel()

    def _conectare(self):
        """Deschide conexiunea cu fisierul DB."""
        return sqlite3.connect(self.nume_db)

    def creeaza_tabel(self):
        """Creeaza tabelul principal cu suport pentru vectori AI si Categorii Smart."""
        conn = self._conectare()
        cursor = conn.cursor()
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
                categorie TEXT,  -- Aici salvam categoria (Oameni, Natura, etc.)
                vector_ai BLOB   -- Amprenta AI salvata ca binar
            )
        ''')
        conn.commit()
        conn.close()

    def salveaza_sau_actualizeaza(self, d):
        """Salveaza datele imaginii, inclusiv vectorul AI si Categoria."""
        conn = self._conectare()
        cursor = conn.cursor()
        
        vector_binar = None
        if d.get('vector_ai') is not None:
            vector_binar = pickle.dumps(d.get('vector_ai'))

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO imagini 
                (cale, nume, format, rezolutie, mb, marca, model, data_poza, gps, cale_cache, categorie, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                d.get('cale'), 
                d.get('nume'), 
                d.get('format'), 
                d.get('rezolutie'), 
                d.get('mb'), 
                d.get('marca'), 
                d.get('model'), 
                d.get('data'), 
                d.get('gps'),
                d.get('cale_cache'),
                d.get('categorie'), 
                vector_binar
            ))
            conn.commit()
        except Exception as e:
            print(f"Eroare SQLite la salvare: {e}")
        finally:
            conn.close()

    def cauta_dupa_cale(self, cale_fisier):
        """Returneaza datele unei imagini specifice."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM imagini WHERE cale = ?", (cale_fisier,))
        date = cursor.fetchone()
        conn.close()
        return date

    def obtine_toti_vectorii(self):
        """Incarca toti vectorii pentru motorul de cautare FAISS."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale, vector_ai FROM imagini WHERE vector_ai IS NOT NULL")
        rezultate = cursor.fetchall()
        conn.close()

        date_ai = []
        for cale, v_binar in rezultate:
            try:
                v_numpy = pickle.loads(v_binar)
                date_ai.append((cale, v_numpy))
            except:
                continue
        return date_ai

    # --- FUNCTII NOI PENTRU SMART ALBUMS ---

    def numara_per_categorie(self, nume_categorie):
        """Intoarce numarul de poze dintr-o anumita categorie AI."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM imagini WHERE categorie = ?", (nume_categorie,))
        rezultat = cursor.fetchone()
        conn.close()
        
        if rezultat:
            return rezultat[0]
        return 0

    def obtine_cai_dupa_categorie(self, nume_categorie):
        """Returneaza lista de cai complete (paths) pentru o anumita categorie AI."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT cale FROM imagini WHERE categorie = ?", (nume_categorie,))
        rezultate = cursor.fetchall()
        conn.close()
        
        # Transformam lista de tuple in lista simpla de cai
        cai_finale = []
        for r in rezultate:
            cai_finale.append(r[0])
            
        return cai_finale