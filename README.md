# Browser-Assisted-Audio-Extraction-and-RAG-Pipeline-
# YouTube Audio-to-Text RAG Assistant

An end-to-end AI pipeline that extracts audio from YouTube videos, converts speech into text using Whisper, generates embeddings from transcripts, and enables question-answering over video content using Retrieval-Augmented Generation (RAG) with a local LLM.

---

# Features

- Browser-assisted YouTube extraction using Playwright
- Cookie/session-based media access handling
- Audio extraction using yt-dlp
- Audio conversion using FFmpeg
- Speech-to-text transcription using Faster-Whisper
- Automatic language detection
- Transcript chunking
- Semantic embeddings generation
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- Local LLM-based question answering
- Runs locally or on Azure VM

---

# Architecture Flow

```text
YouTube URL
    ↓
Playwright Browser Automation
    ↓
Session / Cookie Extraction
    ↓
yt-dlp Audio Download
    ↓
FFmpeg Audio Conversion
    ↓
Whisper Transcription
    ↓
Transcript Generation
    ↓
Text Chunking
    ↓
Sentence Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Local LLM
    ↓
Question Answering
```
## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Browser Automation** | Playwright |
| **Media Extraction** | yt-dlp |
| **Audio Processing** | FFmpeg |
| **Speech Recognition** | Faster-Whisper |
| **Embeddings** | SentenceTransformers |
| **Vector Search** | FAISS |
| **Local LLM** | TinyLlama |
| **Backend Language** | Python |
| **Deployment** | Azure VM |

## Project Structure
```
youtube-rag-assistant/
│
├── app.py
├── chat.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── audio/
├── transcripts/
└── vector_store/
```
## Installation
Clone Repository
```bash
git clone <your-github-repo-url>
cd youtube-rag-assistant
```
Create Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```
Install Dependencies
```bash
pip install -r requirements.txt
```
Install Playwright Browser
```bash
playwright install chromium
```
Install FFmpeg
Ubuntu / Azure VM
```bash
sudo apt update
sudo apt install ffmpeg -y
```
Running the Project
```</>
python app.py
python chat.py
```

## Example Workflow
* Enter YouTube URL
* Browser session is created using Playwright
* yt-dlp downloads audio
* Whisper transcribes speech
* Transcript embeddings are created
* FAISS retrieves relevant transcript chunks
* Local LLM answers user questions

## Requirements
* yt-dlp
* playwright
* faster-whisper
* torch
* sentence-transformers
* transformers
* accelerate
* faiss-cpu
* huggingface_hub
## Key Concepts Implemented
* Browser Automation
* Audio Extraction Pipelines
* Speech Recognition
* Embedding Models
* Vector Databases
* Semantic Search
* Retrieval-Augmented Generation (RAG)
* Local LLM Inference
* Azure VM Deployment
* GPU / CUDA Environment Debugging
## Future Improvements
* Streamlit UI
* FastAPI backend
* Multi-video ingestion
* Persistent vector database
* GPU acceleration
* Docker deployment
* Real-time transcription
* Chat history memory
## Notes
* GPU acceleration requires Azure GPU-enabled VM instances (NC/ND/NV series).
* Standard CPU VMs do not provide CUDA hardware acceleration.
* Whisper transcription performance depends on available compute resources.
