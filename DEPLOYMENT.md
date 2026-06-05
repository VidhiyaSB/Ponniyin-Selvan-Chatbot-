# Deployment Guide

This guide covers how to deploy the Ponniyin Selvan Chatbot to Hugging Face Spaces.

---

## 🚀 Hugging Face Spaces Deployment (via GUI)

### Exact Files/Folders to Upload

> [!IMPORTANT]
> **Upload ONLY these files and folders. Nothing else.**

| ✅ Upload This | Description |
|---|---|
| `app.py` | Main Gradio application |
| `build_index.py` | Index builder script |
| `requirements.txt` | Python dependencies |
| `README.md` | Space description (HF uses this) |
| `src/` folder | **Entire folder** with all `.py` files inside |
| `assets/` folder | **Entire folder** (contains `style.css`) |
| `data/` folder | **Entire folder** (contains `faiss_index/`) |

> [!CAUTION]
> **Do NOT upload these:**
> - `venv/` - Virtual environment (HF creates its own)
> - `.env` - Your secret keys (add via HF Secrets instead)
> - `__pycache__/` - Python cache
> - `.git/` - Git history
> - `.gitignore` - Not needed on HF
> - `.env.example` - Not needed
> - PDF files - Already processed into the index

---

### Step-by-Step GUI Upload

1.  **Create Space**: Go to [huggingface.co/new-space](https://huggingface.co/new-space)
    - SDK: **Gradio**
    - Hardware: **CPU Basic** (free tier works)

2.  **Upload via "Files" tab**:
    - Click **"+ Add file"** → **"Upload files"**
    - Drag and drop the files/folders listed above
    - For folders (`src/`, `assets/`, `data/`), upload them one by one

3.  **Add Secrets** (Settings tab):
    - `MISTRAL_API_KEY` = your Mistral API key
    - `ELEVENLABS_API_KEY` = your ElevenLabs key (optional)

4.  **Done!** The Space will auto-build and run.

---

## 💻 Local Deployment

---

## 💻 Local Deployment

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd ps-chatbot
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
# Edit .env and allow your keys
```

### 4. Build Index (If not present)
If the `data/faiss_index` folder is missing, you need to parse the PDFs and build the index:
```bash
python build_index.py
```
*Note: Ensure the PDF files are in the root directory.*

### 5. Run Application
```bash
python app.py
```
Open your browser at `http://localhost:7861`.
