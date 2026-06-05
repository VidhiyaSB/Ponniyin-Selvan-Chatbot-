"""
Vector Store for Ponniyin Selvan RAG
Handles embeddings and FAISS vector database
"""
import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Run: pip install sentence-transformers faiss-cpu")
    raise


class VectorStore:
    """FAISS-based vector store with multilingual embeddings"""
    
    # Faster multilingual embedding model (~420MB vs 2.2GB)
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(
        self, 
        model_name: str = None,
        index_path: str = None
    ):
        """
        Initialize the vector store
        
        Args:
            model_name: Hugging Face model name for embeddings
            index_path: Path to save/load the FAISS index
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.embedding_model = None
        self.index = None
        self.documents = []  # Store document chunks for retrieval
        
        # Default index path
        if index_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.index_path = os.path.join(base_dir, "data", "faiss_index")
        else:
            self.index_path = index_path
    
    def _load_embedding_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings.astype('float32')
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build FAISS index from document chunks
        
        Args:
            documents: List of dicts with 'text' and 'metadata' keys
        """
        if not documents:
            raise ValueError("No documents provided")
        
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self._get_embeddings(texts)
        
        # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dimension})")
    
    def save_index(self) -> None:
        """Save the FAISS index and documents to disk"""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(self.index_path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save documents metadata
        docs_file = os.path.join(self.index_path, "documents.pkl")
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        index_file = os.path.join(self.index_path, "index.faiss")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            return False
        
        try:
            self.index = faiss.read_index(index_file)
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents with scores
        """
        if self.index is None:
            # Try to load from disk
            if not self.load_index():
                raise ValueError("No index available. Build or load index first.")
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        
        return results


def create_vector_store(documents: List[Dict[str, Any]], save: bool = True) -> VectorStore:
    """
    Convenience function to create and optionally save a vector store
    
    Args:
        documents: List of document chunks
        save: Whether to save the index to disk
        
    Returns:
        Initialized VectorStore
    """
    vs = VectorStore()
    vs.build_index(documents)
    if save:
        vs.save_index()
    return vs


if __name__ == "__main__":
    # Test vector store
    from document_processor import load_documents
    
    # Check if index exists
    vs = VectorStore()
    if not vs.load_index():
        print("Building new index...")
        docs = load_documents()
        vs.build_index(docs)
        vs.save_index()
    
    # Test search
    results = vs.search("Vandiyathevan", top_k=3)
    print("\nSearch results for 'Vandiyathevan':")
    for r in results:
        print(f"Score: {r['score']:.3f}")
        print(f"Volume: {r['metadata']['volume']}")
        print(f"Text: {r['text'][:200]}...")
        print("-" * 50)
