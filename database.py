import sqlite3

class ManagerBazaDate:
    def __init__(self, nume_db="galerie_licenta.db"):
        self.nume_db = nume_db
        self.creeaza_tabel()

    def _conectare(self):
        """Deschide conexiunea cu fisierul DB."""
        return sqlite3.connect(self.nume_db)

    def creeaza_tabel(self):
        """Creeaza tabelul principal unde stocam metadatele si calea catre cache."""
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
                cale_cache TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def salveaza_sau_actualizeaza(self, d):
        """Salveaza dictionarul de date primit de la Worker (include acum si cache)."""
        conn = self._conectare()
        cursor = conn.cursor()
        try:
            # Am adaugat 'cale_cache' atat in lista de coloane cat si un '?' in plus
            cursor.execute('''
                INSERT OR REPLACE INTO imagini 
                (cale, nume, format, rezolutie, mb, marca, model, data_poza, gps, cale_cache)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                d.get('cale_cache')  # Noua valoare salvata
            ))
            conn.commit()
        except Exception as e:
            print(f"Eroare SQLite la salvare: {e}")
        finally:
            conn.close()

    def cauta_dupa_cale(self, cale_fisier):
        """Returneaza datele (inclusiv calea cache) daca poza a mai fost procesata."""
        conn = self._conectare()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM imagini WHERE cale = ?", (cale_fisier,))
        date = cursor.fetchone()
        conn.close()
        return date