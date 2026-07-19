# Fashion Studio ETL Pipeline

Proyek submission ETL sederhana untuk mengekstrak, mentransformasi, dan
menyimpan data produk dari Fashion Studio.

## Status Implementasi

Fase 1 telah selesai:

- Struktur proyek modular tersedia.
- Package `utils` dan `tests` telah dibuat.
- Dependencies proyek tersedia dalam `requirements.txt`.
- Konfigurasi dasar pytest tersedia.
- Template environment dan service account tersedia.

Tahapan berikutnya:

1. Phase 2: Extract.
2. Phase 3: Transform.
3. Phase 4: Load.
4. Phase 5: Main pipeline.
5. Phase 6: Unit test dan coverage.
6. Phase 7: Finalisasi submission.

## Membuat Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux atau macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan Project

```bash
python main.py
```

## Menjalankan Unit Test

```bash
python -m pytest tests
```

## Menjalankan Test Coverage

```bash
python -m pytest tests --cov=utils --cov-report=term-missing
```

## Kredensial

Salin `.env.example` menjadi `.env`, lalu isi konfigurasi lokal.

File `google-sheets-api.example.json` hanya template. Sebelum menggunakan
Google Sheets, sediakan service account asli dengan nama
`google-sheets-api.json`.

Jangan menggunakan kredensial produksi untuk submission.
