"""
Build the vector index from Ponniyin Selvan PDFs
Run this once before starting the chatbot
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.document_processor import load_documents
from src.vector_store import create_vector_store

if __name__ == "__main__":
    print("=" * 50)
    print("Ponniyin Selvan - Building Vector Index")
    print("=" * 50)
    
    print("\n📚 Loading and processing PDFs...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} text chunks")
    
    print("\n🔧 Building FAISS vector index...")
    print("   (This may take a few minutes on first run as the embedding model downloads)")
    vs = create_vector_store(documents, save=True)
    
    print("\n✅ Vector index built and saved successfully!")
    print("   You can now run: python app.py")
