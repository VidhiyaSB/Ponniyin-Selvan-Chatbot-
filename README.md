---
title: Ponniyin Selvan Chatbot
emoji: 🏛️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🏛️ பொன்னியின் செல்வன் | Ponniyin Selvan Chatbot

A bilingual (Tamil/English) RAG chatbot for exploring Kalki Krishnamurthy's epic historical novel about the Chola dynasty.

## Features

- 📚 **RAG-powered**: Answers based on the actual novel text
- 🌐 **Bilingual**: Supports both Tamil and English queries
- 🔊 **Voice Output**: ElevenLabs text-to-speech integration
- 🎨 **Historical Theme**: Chola dynasty inspired UI design

## How to Use

1. Ask questions about characters, plot, or historical context
2. Toggle voice response for audio output
3. View source references from the novel

## Example Questions

- "Who is Vandiyathevan?"
- "வந்தியத்தேவன் யார்?"
- "Tell me about Kundavai"
- "What happens at the end of the story?"

## Built With

- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2 (Sentence Transformers)
- **LLM**: Mistral Large
- **Vector Store**: FAISS
- **Voice**: ElevenLabs (eleven_multilingual_v2)
- **UI**: Gradio

---

*Based on Ponniyin Selvan by Kalki Krishnamurthy (1950-1954)*
