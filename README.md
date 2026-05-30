# SpaceAx AI

Mesin percakapan bahasa Indonesia berbasis Transformer decoder-only, dilatih dari nol di mesin sendiri (bukan model siap pakai dari HuggingFace). Dibangun Thomas Alfareno Ananta Nugraha — Teknik Informatika FTEIC ITS Surabaya, untuk Space Ax Corp.

**Repositori:** [github.com/thomasalfareno/SpaceAX-AI-BETA-TPU](https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU)  
**Versi:** 2.0.0

---

## Isi

1. [Gambaran singkat](#gambaran-singkat)
2. [Folder `kbbi/` & sinkron leksikon](#folder-kbbi--sinkron-leksikon)
3. [Struktur folder & modul](#struktur-folder--modul)
4. [Persyaratan](#persyaratan)
5. [Instalasi Windows](#instalasi-windows)
6. [Instalasi Linux](#instalasi-linux)
7. [TPU v5e-1 (utama — training ProMax)](#tpu-v5e-1-utama--training-promax)
8. [Google Colab (TPU & GPU)](#google-colab-tpu--gpu)
9. [Perintah CLI](#perintah-cli)
10. [ProMax 1B / 4B / 8B](#promax-1b--4b--8b)
11. [Ukuran vocab per profil](#ukuran-vocab-per-profil)
12. [Training & chat: apa yang wajar diharapkan](#training--chat-apa-yang-wajar-diharapkan)
13. [Variabel lingkungan](#variabel-lingkungan)
14. [Masalah umum](#masalah-umum)

---

## Gambaran singkat

Clone proyek:

```bash
git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
cd SpaceAX-AI-BETA-TPU
```

Alur kerja biasa:

1. Clone repo → **ekstrak `kbbi/ekstrak.zip` sekali** (isi JSON KBBI + leksikon txt).
2. **Training (disarankan):** runtime **TPU v5e-1** + `pip install -r requirements-colab-tpu.txt` (Colab) → `python main.py verify-tpu` → `python main.py train`.
3. **Lokal / GPU:** `pip install -r requirements.txt -r requirements-torch.txt` → `python main.py train`.
4. `python main.py chat` — ngobrol; model mencoba generate teks sendiri dulu, baru fallback jika output tidak layak.
5. (Opsional) `python main.py retrain` — gabung log chat ke dataset lalu latih ulang.

Fitur utama: memori percakapan (STM/LTM), mesin emosi, **KBBI + leksikon lengkap**, pencarian internet (`ddgs`), augmentasi anti-hafalan, skala **ProMax** (~1.2B / ~4B / ~8B), training dioptimalkan untuk **Google TPU v5e-1** (PyTorch/XLA, bfloat16 native), fallback **CUDA/CPU**.

---

## Folder `kbbi/` & sinkron leksikon

### Wajib sekali: ekstrak `ekstrak.zip`

Di repo ada **`kbbi/ekstrak.zip`** (~10 MB). File JSON/txt KBBI **tidak** dipakai langsung — user harus ekstrak dulu ke folder `kbbi/` (hanya saat instalasi pertama).

Setelah ekstrak, `kbbi/` harus berisi antara lain `kbbi_v_part1.json` … `part4.json` dan file `.txt` leksikon. Baru jalankan `python main.py train`.

**Linux / Colab:**

```bash
unzip -q kbbi/ekstrak.zip -d kbbi/temp
mv kbbi/temp/* kbbi/
rm -rf kbbi/temp kbbi/ekstrak.zip
```

**Windows (PowerShell):**

```powershell
Expand-Archive -Path kbbi\ekstrak.zip -DestinationPath kbbi\temp
Move-Item kbbi\temp\* kbbi\
Remove-Item kbbi\temp -Recurse -Force
Remove-Item kbbi\ekstrak.zip
```

Cek cepat: `ls kbbi/kbbi_v_part1.json` (Linux) atau `dir kbbi\kbbi_v_part1.json` (Windows).

---

Semua file di `kbbi/` dipakai otomatis oleh `core/kbbi.py`:

| File | Fungsi |
|------|--------|
| `ekstrak.zip` | Arsip instalasi — **ekstrak sekali**, lalu boleh dihapus |
| `kbbi_v_part1.json` … `part4.json` | Definisi KBBI resmi (~112k entri) |
| `indonesian-words.txt` | Daftar kata Indonesia |
| `list_0.5.1.txt`, `list_1.0.0.txt` | Daftar kata tambahan (besar) |
| `combined_slang_words.txt` | JSON gaul→baku (mis. `gw` → `saya`) |
| `combined_root_words.txt` | Kata dasar |
| `combined_stop_words.txt` | Partikel / stop word (kaidah tata bahasa) |

**Ke mana datanya masuk:**

1. **Seed training** (`data/seed/conversations.json`) — ribuan pasangan: definisi, slang, leksikon, grammar.
2. **Tokenizer BPE** — corpus definisi + slang + sampel kosakata (~14 juta karakter).
3. **Chat** — tanya arti kata KBBI atau arti gaul (`apa arti gw`).

**Sinkron otomatis** saat `train` jika file di `kbbi/` lebih baru dari seed, atau jumlah pasangan `kbbi_*` di seed masih sedikit. **Paksa ulang:**

```bash
export SPACEAX_KBBI_SYNC=1
python main.py train --regen
```

Setelah menambah file KBBI baru, hapus checkpoint & vocab lama lalu `--regen` (ukuran embedding berubah jika vocab profil berubah).

---

## Struktur folder & modul

```
SpaceAX-AI-BETA-TPU/            # nama folder setelah git clone
├── main.py                 # CLI: train, chat, learn, retrain, test, chatdev
├── chat.py                 # UI terminal + generasi + fallback percakapan
├── requirements.txt            # Paket aplikasi (tanpa torch)
├── requirements-torch.txt      # PyTorch CPU/CUDA (lokal)
├── requirements-colab-tpu.txt  # Colab TPU — jangan turunkan torch
├── requirements-tpu.txt        # GCE / TPU VM
├── scripts/
│   ├── install_colab_tpu.py    # Instal Colab (disarankan)
│   ├── install_tpu.sh          # Instal TPU (deteksi Colab otomatis)
│   └── verify_tpu.py           # Tes PyTorch/XLA
├── core/
│   ├── accelerator.py      # TPU / CUDA / CPU — device, mark_step, diagnostik
│   ├── config.py           # Profil model (small→promax), training, deteksi RAM/HBM
│   ├── promax.py           # Sub-tier promax_1b / 4b / 8b
│   ├── vram_fit.py         # Mem-fit ProMax 8B untuk HBM/VRAM
│   ├── model.py            # SpaceaxModel — Transformer decoder-only + RoPE
│   ├── tokenizer.py        # BPETokenizer (tokenizers HuggingFace)
│   ├── kbbi.py             # KBBI JSON + txt slang/list/root → seed & tokenizer
│   └── debug_log.py        # Log opsional (SPACEAX_DEBUG=1) → data/logs/
├── training/
│   ├── generate_seed_data.py   # conversations.json (math, emosi, dll.)
│   ├── seed_extra.py           # Topik tambahan (teknologi, budaya, coding massal)
│   ├── composition_variants.py # Variasi intent (anti hafal satu kalimat)
│   ├── text_augment.py         # Paraphrase on-the-fly
│   ├── dataset.py              # DataLoader + augmentasi dinamis
│   └── trainer.py              # Loop training, checkpoint, sample per epoch
├── learning/
│   ├── internet.py         # Pencarian & cache (ddgs)
│   ├── web_learner.py        # Belajar topik dari web
│   ├── knowledge_base.py     # Penyimpanan pengetahuan terstruktur
│   └── auto_trainer.py       # Hook auto-retrain (jika dipakai)
├── memory/
│   ├── memory.py             # STM buffer + LTM SQLite
│   └── vector_store.py       # Pencarian fakta mirip (embedding sederhana)
├── personality/
│   └── emotion_engine.py     # 9 emosi + decay + pengaruh gaya jawaban
├── data/
│   ├── seed/conversations.json
│   ├── checkpoints/model_best.pt, model_epoch_N.pt
│   ├── vocab/                # BPE tersimpan
│   ├── knowledge/              # Hasil internet
│   ├── memories/               # Memori & gaya user
│   └── conversation_logs/      # chat_history.json untuk retrain
└── kbbi/
    ├── ekstrak.zip           # ekstrak sekali saat instalasi → file di bawah
    ├── kbbi_v_part1.json …   # (hasil ekstrak)
    └── …                     # slang, list kata, root, stop words
```

### Peran modul (ringkas)

| Modul | Fungsi |
|-------|--------|
| `core/model.py` | Arsitektur neural; `generate()` untuk inferensi token demi token. |
| `core/kbbi.py` | Muat leksikon, `enrich_all_training_data()`, `generate_corpus()`, deteksi gibberish. |
| `core/config.py` | Auto-pilih ukuran model dari RAM; override `--size`, `--promax-tier`, `--force`. |
| `training/trainer.py` | AdamW/Adafactor, AMP, early stopping (bisa dimatikan dengan `--force`). |
| `chat.py` | Routing: matematika → KBBI/slang → knowledge → **model** → internet; validator 3 tingkat. |
| `personality/emotion_engine.py` | Deteksi emosi; ProMax: mood lebih halus + `max_gen_len` lebih panjang. |
| `memory/memory.py` | Konteks beberapa giliran terakhir masuk prompt model. |

---

## Persyaratan

### Umum

| Komponen | Minimum | Disarankan (ProMax) |
|----------|---------|---------------------|
| Python | 3.10 | 3.10–3.11 (runtime Colab TPU) |
| RAM host | 8 GB | 48 GB+ (init tokenizer + data) |
| Disk | 5 GB bebas | 20 GB+ (checkpoint 8B) |
| Jaringan | Opsional | Untuk `learn` / `ddgs` |

- Setelah ekstrak `kbbi/ekstrak.zip`: `kbbi_v_part1.json` … `part4.json` + file txt leksikon.
- File dependensi: lihat tabel di bawah (pilih sesuai lingkungan).

### File requirements

| File | Kapan dipakai |
|------|----------------|
| `requirements.txt` | Selalu — numpy, tokenizers, ddgs, rich, … (**tanpa** torch) |
| `requirements-torch.txt` | PC lokal / server **GPU atau CPU** |
| `requirements-colab-tpu.txt` | **Google Colab + runtime TPU v5e-1** |
| `requirements-tpu.txt` | **GCE / TPU VM** (bukan Colab) |

| Paket (requirements.txt) | Fungsi |
|--------------------------|--------|
| `numpy`, `tokenizers` | Data & BPE |
| `ddgs`, `rich`, `requests`, `beautifulsoup4` | Chat & web |

| Paket (TPU) | Fungsi |
|-------------|--------|
| `torch_xla[tpu]>=2.8.0` | PyTorch/XLA (index libtpu; **bukan** `<2.6`) |
| Index `libtpu-releases` | Wheel resmi Google |

**Colab:** jangan `pip install torch` dari pin lama — runtime sudah punya torch 2.9.x; memaksa 2.6 memecah `torch_xla`.

Instalasi: `python scripts/install_colab_tpu.py` atau `!pip install -r requirements-colab-tpu.txt`

### Training di TPU v5e-1 (disarankan)

| Komponen | Spesifikasi |
|----------|-------------|
| Perangkat | **1× TPU v5e** (konfigurasi **v5e-1**, 1 chip) |
| Memori akselerator | **16 GB HBM** per chip |
| Runtime | Google Colab (TPU) atau Compute Engine + TPU VM |
| Stack ML | PyTorch ≥ 2.2 + **torch_xla** (`requirements-tpu.txt`) |
| Presisi | **bfloat16** native (`XLA_USE_BF16=1`, otomatis) |

### Training di GPU (alternatif)

| Komponen | Spesifikasi |
|----------|-------------|
| NVIDIA GPU | CUDA; VRAM sesuai tier (lihat [ProMax](#promax-1b--4b--8b)) |
| Stack | `requirements.txt` + PyTorch dengan dukungan CUDA dari [pytorch.org](https://pytorch.org/get-started/locally/) |

### Chat / inferensi

- `python main.py chat` memuat checkpoint di **CPU** (ringan, tidak membutuhkan TPU saat ngobrol).
- Checkpoint hasil training TPU/CUDA/GPU kompatibel (format `.pt` sama).

---

## Instalasi Windows

1. Pasang Python dari [python.org](https://www.python.org/) — centang **Add Python to PATH**.
2. Buka Command Prompt atau PowerShell, masuk ke folder proyek:

```cmd
git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
cd SpaceAX-AI-BETA-TPU
```

3. Ekstrak KBBI (sekali):

```powershell
Expand-Archive -Path kbbi\ekstrak.zip -DestinationPath kbbi\temp
Move-Item kbbi\temp\* kbbi\
Remove-Item kbbi\temp -Recurse -Force
Remove-Item kbbi\ekstrak.zip
```

4. Virtual environment:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt -r requirements-torch.txt
pip install "ddgs>=9.14.0"
```

5. PyTorch + CUDA (jika ada NVIDIA): ganti `requirements-torch.txt` dengan wheel CUDA dari [pytorch.org](https://pytorch.org/get-started/locally/) jika perlu.

6. Cek:

```cmd
python -c "from ddgs import DDGS; import torch; print('ok', torch.__version__)"
```

---

## Instalasi Linux

```bash
git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
cd SpaceAX-AI-BETA-TPU

# Ekstrak KBBI (wajib sekali)
unzip -q kbbi/ekstrak.zip -d kbbi/temp
mv kbbi/temp/* kbbi/
rm -rf kbbi/temp kbbi/ekstrak.zip

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-torch.txt
pip install "ddgs>=9.14.0"
```

Di Arch/Ubuntu dengan PEP 668, pakai venv seperti di atas agar tidak bentrok paket sistem.

Verifikasi:

```bash
python -c "from ddgs import DDGS; import torch; print('ok', torch.__version__)"
```

---

## TPU v5e-1 (utama — training ProMax)

Repositori ini **mengutamakan TPU v5e-1** melalui [PyTorch/XLA](https://cloud.google.com/tpu/docs/pytorch-xla-setup). CUDA tetap didukung sebagai fallback.

### Arsitektur perangkat lunak

```
main.py train
    → core/config.py          (profil model, batch, mem-fit)
    → core/accelerator.py     (deteksi TPU > CUDA > CPU)
    → core/vram_fit.py        (ProMax 8B: seq/batch/accum untuk 16 GB HBM)
    → training/trainer.py     (bf16, xm.mark_step() tiap optimizer step)
```

| Backend | Deteksi | Presisi training | Catatan |
|---------|---------|------------------|---------|
| `tpu` | `torch_xla` + `PJRT_DEVICE=TPU` | bfloat16 | **Disarankan** untuk ProMax |
| `cuda` | `torch.cuda.is_available()` | bf16/fp16 | Workstation / Colab GPU |
| `cpu` | fallback | fp32 | Sangat lambat; ProMax 8B **ditolak** |

### Instalasi cepat (Colab / GCE)

```bash
# 1) Pastikan runtime TPU v5e-1 aktif (bukan GPU/CPU)
export PJRT_DEVICE=TPU
export SPACEAX_ACCELERATOR=tpu

# 2) Colab (disarankan)
python scripts/install_colab_tpu.py
# atau:
pip install -r requirements-colab-tpu.txt

# 2b) GCE / TPU VM (bukan Colab)
pip install -r requirements.txt
pip install -r requirements-tpu.txt

# 3) Verifikasi
python main.py verify-tpu
```

### Training ProMax di TPU v5e-1

```bash
# Tier 1B — paling stabil di 16 GB HBM
python main.py train --size promax --promax-tier promax_1b --epochs 30 --regen

# Tier 8B — mem-fit otomatis (seq_len/batch/bf16); wajib TPU/GPU
python main.py train --size promax --promax-tier promax_8b --epochs 30 --force --regen
```

**Perilaku otomatis di TPU:**

- `mem-fit` menyesuaikan `max_seq_len`, `batch_size`, `gradient_accumulation_steps` untuk ~16 GB HBM.
- Bobot model **bfloat16** (optimal di v5e).
- Optimizer **Adafactor** + gradient checkpointing untuk ProMax.
- `torch_xla.core.xla_model.mark_step()` setelah setiap langkah optimizer (wajib XLA).
- `pin_memory=False` di DataLoader (tidak relevan untuk TPU).

### Matriks versi (referensi)

| Lingkungan | Python | PyTorch | torch_xla |
|------------|--------|---------|-----------|
| Colab TPU v5e (2025–2026) | 3.11–3.12 | **2.9.x** (bawaan Colab) | `requirements-colab-tpu.txt` → torch_xla **2.8+** |
| GCE TPU VM | 3.10–3.11 | Selaras image | `requirements-tpu.txt` |

**Jangan** memasang `torch<2.7` di Colab — error umum:

`No matching distribution found for torch_xla<2.6.0`

Gunakan `requirements-colab-tpu.txt` (tanpa pin torch lama).

### Google Cloud Engine (ringkas)

1. Buat VM dengan akselerator **TPU v5e-1** (1 chip) di zona yang mendukung v5e.
2. SSH ke VM worker TPU, clone repo, ekstrak `kbbi/`.
3. `bash scripts/install_tpu.sh`
4. `python main.py train --size promax ...`
5. Simpan `data/checkpoints/model_best.pt` ke Cloud Storage jika perlu.

Detail resmi: [Cloud TPU — PyTorch](https://cloud.google.com/tpu/docs/run-calculation-jax).

---

## Google Colab (TPU & GPU)

### Colab — TPU v5e-1 (disarankan)

Urutan: **runtime TPU** → clone → ekstrak KBBI → install TPU → verify → train.

```python
# 0) Runtime: menu Runtime → Ubah jenis runtime → TPU v5e-1 → Simpan

import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["SPACEAX_ACCELERATOR"] = "tpu"

# 1) Clone
!git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
%cd SpaceAX-AI-BETA-TPU

# 2) Ekstrak KBBI (wajib sekali)
!unzip -q kbbi/ekstrak.zip -d kbbi/temp
!mv kbbi/temp/* kbbi/
!rm -rf kbbi/temp kbbi/ekstrak.zip

# 3) Dependensi TPU (JANGAN pip install -r requirements-tpu.txt di Colab)
!pip install -q -U pip
!pip install -q -r requirements-colab-tpu.txt
# atau satu langkah:
# !python scripts/install_colab_tpu.py

# 4) Verifikasi TPU
!python main.py verify-tpu

# 5) Training ProMax 1B (stabil di 16 GB HBM)
!python main.py train --size promax --promax-tier promax_1b --epochs 30 --regen
```

**ProMax 8B di Colab TPU v5e-1** (16 GB HBM — mem-fit otomatis):

```python
!python main.py train --size promax --promax-tier promax_8b --epochs 30 --force --regen
```

Simpan checkpoint ke Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/SpaceAX/checkpoints"
!cp data/checkpoints/model_best.pt "/content/drive/MyDrive/SpaceAX/checkpoints/"
```

Chat setelah training (CPU, di notebook yang sama):

```python
!python main.py chat --size promax --promax-tier promax_1b
```

### Colab — GPU (alternatif)

Contoh sel di notebook (urutan penting: clone → **ekstrak zip** → pip → train):

```python
# 1) Mount Drive (opsional, untuk simpan checkpoint)
from google.colab import drive
drive.mount('/content/drive')

# 2) Clone repo
!git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
%cd SpaceAX-AI-BETA-TPU

# 3) Ekstrak KBBI (wajib sekali — dari kbbi/ekstrak.zip di repo)
!unzip -q kbbi/ekstrak.zip -d kbbi/temp
!mv kbbi/temp/* kbbi/
!rm -rf kbbi/temp kbbi/ekstrak.zip

# 4) Dependensi
!pip install -q -U pip
!pip install -q -r requirements.txt
!pip install -q "ddgs>=9.14.0"

# 5) Cek KBBI terbaca (opsional)
!ls kbbi/kbbi_v_part1.json

# 6) Tier aman untuk T4 16GB
import os
os.environ["SPACEAX_PROMAX_TIER"] = "promax_1b"

# 7) Training
!python main.py train --size promax --regen --epochs 30
```

Simpan checkpoint ke Drive:

```python
!mkdir -p "/content/drive/MyDrive/SpaceAX/checkpoints"
!cp data/checkpoints/model_best.pt "/content/drive/MyDrive/SpaceAX/checkpoints/"
```

### ProMax 8B di Colab (A100 40 GB+)

**Jangan** di runtime T4 (15 GB VRAM) — akan macet/OOM di `Inisialisasi Model`. Pilih *Runtime → A100*.

```python
!git clone https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU.git
%cd SpaceAX-AI-BETA-TPU

!unzip -q kbbi/ekstrak.zip -d kbbi/temp
!mv kbbi/temp/* kbbi/
!rm -rf kbbi/temp kbbi/ekstrak.zip

!pip install -q -r requirements.txt
!pip install -q "ddgs>=9.14.0"

import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# Tes 2 epoch (chat belum pintar; naikkan epoch untuk hasil nyata)
!python main.py train --size promax --promax-tier promax_8b --epochs 2 --batch-size 1 --grad-accum 16 --regen --force
```

`--force` = tier **tetap 8B** (tidak downgrade), early stopping mati, plus **VRAM-fit** otomatis (`seq_len` / batch / bobot bf16 disesuaikan GPU).  
`--epochs 2` hanya untuk cek pipeline jalan; chat baru enak setelah puluhan epoch.

---

## Perintah CLI

Semua lewat:

```bash
python main.py <perintah> [opsi]
```

### `train` — melatih model

```bash
python main.py train
```

| Opsi | Keterangan |
|------|------------|
| `--size` | `small`, `medium`, `large`, `ultra`, `promax` (default: auto dari RAM) |
| `--promax-tier` | `promax_1b`, `promax_4b`, `promax_8b` |
| `--epochs` | Jumlah epoch (ProMax default minimal 30) |
| `--batch-size` | Batch per langkah |
| `--grad-accum` | Akumulasi gradien (batch efektif = batch × accum) |
| `--regen` | Buat ulang seed + tokenizer + sinkron KBBI/leksikon |
| `--force` | **Tidak** turunkan tier/RAM; **matikan** early stopping; jalankan semua epoch |

### `verify-tpu` — cek TPU sebelum training

```bash
python main.py verify-tpu
```

Menjalankan tes alokasi tensor, matmul bfloat16, `nn.Linear`, dan `mark_step()`. Colab: setelah `requirements-colab-tpu.txt`. GCE: setelah `requirements-tpu.txt`.

**Contoh**

```bash
# Profil menengah, 40 epoch
python main.py train --size medium --epochs 40

# ProMax 1B (TPU v5e-1 / GPU)
python main.py train --size promax --promax-tier promax_1b --epochs 30

# ProMax 8B di TPU v5e-1 (mem-fit HBM 16 GB otomatis)
python main.py train --size promax --promax-tier promax_8b --epochs 50 --force --regen

# ProMax 8B dipaksa walau RAM/HBM ketat
python main.py train --size promax --promax-tier promax_8b --epochs 50 --force --batch-size 1 --grad-accum 24

# Setelah menambah file di kbbi/ atau ubah vocab
export SPACEAX_KBBI_SYNC=1
python main.py train --regen

# Hanya regenerasi seed tanpa KBBI paksa (jika sudah sinkron)
python main.py train --regen
```

Output: `data/checkpoints/model_best.pt`, `data/checkpoints/model_epoch_<N>.pt`.

---

### `chat` — percakapan interaktif

```bash
python main.py chat
python main.py chat --mode chatdev
python main.py chat --size promax --promax-tier promax_1b
python chat.py
```

Di dalam chat:

- `!search <topik>` — cari internet & simpan ke knowledge base.
- Riwayat beberapa giliran ikut ke prompt model (bukan hanya pesan terakhir).

**Mode chatdev:** debugging / introspeksi training (perintah internal tambahan di sesi).

---

### `learn` — belajar satu topik dari web

```bash
python main.py learn "transformer neural network"
python main.py learn "hukum archimedes"
```

Butuh `ddgs` dan koneksi jaringan.

---

### `retrain` — latih ulang dengan log percakapan

Setelah beberapa kali `chat`, log ada di `data/conversation_logs/chat_history.json`.

```bash
python main.py retrain
python main.py retrain --size promax --epochs 25 --promax-tier promax_1b
python main.py retrain --force --epochs 40
```

Retrain menggabungkan log ke `data/seed/conversations.json`, menghapus vocab lama, lalu memanggil pipeline `train` lagi.

---

### `test` — uji modul tanpa chat panjang

```bash
python main.py test
```

Menjalankan beberapa pertanyaan uji (identitas, kode, pengetahuan, emosi).

---

### `chatdev` — alias chat mode dev

```bash
python main.py chatdev
```

Sama dengan `python main.py chat --mode chatdev`.

---

## ProMax 1B / 4B / 8B

ProMax = arsitektur SpaceAx dengan skala berbeda (bukan unduh LLM eksternal).

| Tier | ~Parameter | Vocab | RAM host | HBM/VRAM training |
|------|------------|-------|----------|-------------------|
| `promax_1b` | ~1.2B | 96k | 48 GB | 16 GB (TPU v5e-1, T4) |
| `promax_4b` | ~4B | 128k | 64 GB | ≥24 GB |
| `promax_8b` | ~8B | 160k | 96 GB | 16 GB+ dengan **mem-fit** (TPU v5e-1 bf16) atau ≥40 GB GPU |

Profil `small`–`ultra` juga memakai vocab lebih besar (72k–128k). Setelah ubah vocab, wajib `train --regen`.

Pemilihan otomatis (tanpa paksa): RAM ≥96 & HBM/VRAM ≥16 GB → 8B; RAM ≥64 → 4B; selain itu → 1B.

Paksa tier:

```bash
export SPACEAX_PROMAX_TIER=promax_8b
python main.py train --size promax --force
```

atau:

```bash
python main.py train --size promax --promax-tier promax_8b --force
```

**`--force` + ProMax 8B:** tier tidak diturunkan; **mem-fit** (`core/vram_fit.py`) menyesuaikan `max_seq_len`, batch, grad accum, dan bobot **bfloat16** di TPU v5e-1 / GPU &lt;40 GB. Di 16 GB HBM, `seq_len` sering 256–512 — arsitektur tetap 8B, bukan downgrade tier.

Checkpoint **tidak** bisa dipakai antar tier (ukuran layer & vocab beda).

---

## Ukuran vocab per profil

| Profil | Vocab BPE (target) |
|--------|-------------------|
| `small` | 72.000 |
| `medium` / `large` | 96.000 |
| `ultra` | 128.000 |
| `promax_1b` | 96.000 |
| `promax_4b` | 128.000 |
| `promax_8b` | 160.000 |

Tokenizer memakai `min_frequency=1` untuk vocab ≥ 96k agar lebih banyak token unik dari corpus KBBI.

Dataset seed dasar ~4.000+ pasangan (`generate_seed_data` + `seed_extra`); saat training ditambah **~8.000–12.000** pasangan KBBI/leksikon tergantung profil.

---

## Training & chat: apa yang wajar diharapkan

### Training

- **Epoch 1** hampir selalu menghasilkan `val_loss` tinggi (6–8+ pada ProMax). Itu normal; model baru belajar distribusi token.
- **Early stopping** (default): berhenti jika val loss tidak membaik beberapa epoch berturut-turut. Gunakan `--force` agar **semua** epoch di `--epochs` / default ProMax (≥30) tetap jalan.
- Batch besar menstabilkan gradien, tetapi **tidak** menggantikan epoch yang cukup.

### Chat terasa "template"

Penyebab umum:

1. Baru **1 epoch** — otak belum cukup; lanjutkan sampai `val_loss` turun (target nyaman ~4, ideal ~3.5).
2. Checkpoint tier beda dengan yang dilatih — load `model_best.pt` yang cocok dengan `--promax-tier` saat chat.
3. Output model ditolak validator → sekarang chat mencoba 3 tingkat (ketat → longgar → draft) sebelum fallback; setelah epoch sedikit, Anda akan melihat teks dari model meski masih kaku.

| val_loss (di checkpoint) | Perilaku |
|--------------------------|----------|
| ≤ 3.5 | Generasi model prioritas, validator ketat |
| 3.5 – 5.5 | Model + validator longgar |
| 5.5 – 7.5 | Model + validator draft (teks mentah lebih sering lolos) |
| > 7.5 | Tetap dicoba dengan validator draft; fallback jika gagal |

File yang dimuat chat: `data/checkpoints/model_best.pt`.

---

## Variabel lingkungan

| Variabel | Fungsi |
|----------|--------|
| `SPACEAX_ACCELERATOR` | `auto` (default: TPU jika ada), `tpu`, `cuda`, `cpu` |
| `PJRT_DEVICE` | Set ke `TPU` di runtime TPU (otomatis jika terdeteksi Colab/GCE) |
| `SPACEAX_TPU_HBM_GB` | Override ukuran HBM untuk mem-fit (default `16` = v5e-1) |
| `SPACEAX_TPU_CHIPS` | Override jumlah chip (default dari `xla_world_size()`) |
| `XLA_USE_BF16` | `1` — bfloat16 di TPU (diset otomatis) |
| `SPACEAX_PROMAX_TIER` | `promax_1b`, `promax_4b`, `promax_8b` |
| `SPACEAX_FORCE` | `1` / `true` — sama efeknya dengan `--force` |
| `SPACEAX_KBBI_SYNC` | `1` / `true` — paksa gabung ulang KBBI+leksikon ke seed |
| `SPACEAX_DEBUG` | `1` — tulis log ke `data/logs/spaceax_runtime.ndjson` |

---

## Masalah umum

### `No matching distribution found for torch_xla<2.6.0`

Penyebab: `requirements-tpu.txt` versi lama atau `pip install -r requirements.txt` yang memaksa **torch 2.6** di Colab (padahal index libtpu hanya punya **torch_xla 2.8+**).

**Perbaikan di Colab (runtime TPU v5e-1):**

`torch_xla` terpasang di pip tetapi `import` gagal jika **torch masih 2.6+cu124** (sisa runtime GPU) sementara **torch_xla 2.9** butuh **torch 2.9** dari index libtpu.

```python
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["SPACEAX_ACCELERATOR"] = "tpu"

%cd SpaceAX-AI-BETA-TPU   # folder yang berisi main.py

# Pasang ulang torch+xla berpasangan + verifikasi di proses baru
!python scripts/install_colab_tpu.py
```

Jika verify masih gagal: **Runtime → Restart session** (tetap TPU v5e-1) → jalankan ulang **hanya** sel `install_colab_tpu.py` di atas.

Manual (setara):

```python
!pip install -q -U --force-reinstall \
  --extra-index-url https://storage.googleapis.com/libtpu-releases/index.html \
  "torch==2.9.0" "torch_xla[tpu]==2.9.0"
!pip install -q -r requirements.txt
# Restart session, lalu:
!python main.py verify-tpu
```

### TPU tidak terdeteksi / `verify-tpu` gagal

1. **Runtime salah** — Colab harus **TPU v5e-1**, bukan GPU atau CPU. Restart runtime setelah ganti.
2. **`torch_xla` belum terpasang (Colab):**
   ```bash
   pip install -r requirements-colab-tpu.txt
   ```
3. **`PJRT_DEVICE`:** harus `TPU` sebelum import torch_xla:
   ```bash
   export PJRT_DEVICE=TPU
   export SPACEAX_ACCELERATOR=tpu
   python main.py verify-tpu
   ```
4. **torch / torch_xla tidak selaras** — di Colab gunakan `requirements-colab-tpu.txt`, bukan pin `torch<2.7`.

### Training lambat di TPU tapi tidak error

- Epoch pertama mengompilasi graph XLA (normal, bisa 5–15 menit).
- Pastikan `mark_step()` dipanggil (sudah di `training/trainer.py`).
- Kurangi `num_workers` jika host RAM penuh (default 0–2).

### ProMax 8B ditolak di CPU

```
❌ ProMax 8B membutuhkan TPU v5e-1 atau GPU
```

Aktifkan TPU atau GPU; 8B tidak di-init di RAM CPU saja.

### OOM / memori penuh di TPU v5e-1

- Turunkan tier: `--promax-tier promax_1b`
- Pakai mem-fit: `--force` dengan 8B (seq_len otomatis turun)
- `--batch-size 1 --grad-accum 32` atau lebih besar

### `ModuleNotFoundError: No module named 'ddgs'`

```bash
pip install "ddgs>=9.14.0"
```

Pastikan venv aktif: `which python` → path ke `.venv`.

### Macet di `🏗️ Inisialisasi Model Transformer...` (Colab / TPU)

Bukan error — PyTorch/XLA sedang **mengalokasikan bobot** ke HBM/VRAM. Tidak ada progress bar.

- **Jangan Ctrl+C** dulu; ProMax 1B di TPU v5e-1 biasanya **2–8 menit**; 8B bisa **5–15 menit**.
- **TPU:** Runtime → **TPU v5e-1** → `python main.py verify-tpu` harus sukses.
- **GPU:** Runtime → T4/A100 → `torch.cuda.is_available()` harus `True`.
- `promax_8b` di TPU v5e-1: `--force` (mem-fit otomatis); `seq_len` ~256–512. Alternatif ringan:

```bash
!python main.py train --size promax --promax-tier promax_1b --epochs 30 --regen
```

Atau lebih ringan untuk uji cepat:

```bash
!python main.py train --size medium --epochs 20 --regen
```

### CUDA / TPU OOM (memori penuh)

- Turunkan tier: `--promax-tier promax_1b`
- `--batch-size 1 --grad-accum 16` (atau lebih besar accum)
- Profil lebih kecil: `--size large` atau `medium`
- TPU: pastikan `SPACEAX_TPU_HBM_GB=16` (default v5e-1)

### Training ProMax 8B berhenti sendiri di epoch kecil

Tanpa `--force`, early stopping bisa menghentikan training. Jalankan:

```bash
python main.py train --size promax --promax-tier promax_8b --epochs 50 --force
```

### Chat selalu kalimat siap (fallback)

- Lanjutkan training; cek `val_loss` di log training.
- Pastikan `model_best.pt` ada.
- Tier saat chat = tier saat train.

### Checkpoint tidak cocok

Hapus `data/checkpoints/*` dan latih ulang, atau `--regen` jika vocab berubah.

### KBBI tidak terbaca / seed tidak bertambah

- Sudah **ekstrak** `kbbi/ekstrak.zip`? Tanpa itu tidak ada `kbbi_v_part1.json`.
- Pastikan `kbbi/kbbi_v_part1.json` ada: `ls kbbi/kbbi_v_part1.json`
- Cek log training: harus muncul `Leksikon dimuat: … kata unik`.
- Paksa sinkron seed: `SPACEAX_KBBI_SYNC=1 python main.py train --regen`.

---

## Kontak

**Thomas Alfareno Ananta Nugraha**  
Teknik Informatika — FTEIC — ITS Surabaya  
Space Ax Corp — SpaceAx AI  

GitHub: [thomasalfareno/SpaceAX-AI-BETA-TPU](https://github.com/thomasalfareno/SpaceAX-AI-BETA-TPU)
