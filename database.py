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
        """Creeaza tabelul principal cu suport pentru vectori AI."""
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
                vector_ai BLOB  -- Aici salvam amprenta AI (vectorul de 512 numere)
            )
        ''')
        conn.commit()
        conn.close()

    def salveaza_sau_actualizeaza(self, d):
        """Salveaza datele imaginii, inclusiv vectorul AI."""
        conn = self._conectare()
        cursor = conn.cursor()
        
        # Pregatim vectorul AI: daca exista, il transformam in bytes folosind pickle
        vector_binar = None
        if d.get('vector_ai') is not None:
            # d['vector_ai'] este un array numpy; il transformam in format binar
            vector_binar = pickle.dumps(d.get('vector_ai'))

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO imagini 
                (cale, nume, format, rezolutie, mb, marca, model, data_poza, gps, cale_cache, vector_ai)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                vector_binar  # Salvat ca BLOB
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
        
        # Daca avem date si vectorul AI este prezent, ar fi util sa il despachetam,
        # dar de obicei il folosim doar la cautarea globala.
        return date

    def obtine_toti_vectorii(self):
        """
        Incarca toti vectorii salvati pentru a popula indexul FAISS la pornire.
        Returneaza o lista de (cale, vector_numpy).
        """
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