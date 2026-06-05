"""
Document Processor for Ponniyin Selvan PDFs
Handles PDF loading, text extraction, and chunking
"""
import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process Ponniyin Selvan PDF volumes into chunks for RAG"""
    
    VOLUME_NAMES = {
        "New-Floods": "Volume 1: New Floods (புது வெள்ளம்)",
        "The-Storm": "Volume 2: The Storm (சுழற்காற்று)",
        "The-Sword": "Volume 3: The Sword (கொலை வாள்)",
        "The-Crown": "Volume 4: The Crown (மணிமகுடம்)",
        "Epitome-of-Sacrifice": "Volume 5: Epitome of Sacrifice (தியாக சிகரம்)"
    }
    
    def __init__(self, pdf_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            pdf_dir: Directory containing the PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        if pdf_dir is None:
            # Default to the directory containing this script's parent
            self.pdf_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.pdf_dir = pdf_dir
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _get_volume_name(self, filename: str) -> str:
        """Extract volume name from filename"""
        for key, value in self.VOLUME_NAMES.items():
            if key in filename:
                return value
        return "Unknown Volume"
    
    def _extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file with page-level metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with text and metadata
        """
        documents = []
        filename = os.path.basename(pdf_path)
        volume_name = self._get_volume_name(filename)
        
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": filename,
                            "volume": volume_name,
                            "page": page_num,
                            "total_pages": len(reader.pages)
                        }
                    })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
        return documents
    
    def _get_volume_number(self, filename: str) -> int:
        """Extract volume number from filename for sorting"""
        import re
        match = re.search(r'Volume-(\d+)', filename)
        return int(match.group(1)) if match else 99
    
    def load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        """
        Load all PDFs and split into chunks
        
        Returns:
            List of document chunks with metadata
        """
        all_chunks = []
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_dir}")
        
        # Sort by volume number (1, 2, 3, 4, 5)
        pdf_files = sorted(pdf_files, key=self._get_volume_number)
        
        print(f"Found {len(pdf_files)} PDF volumes")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            print(f"Processing: {pdf_file}")
            
            # Extract text with metadata
            page_documents = self._extract_text_from_pdf(pdf_path)
            
            # Chunk each page
            for doc in page_documents:
                chunks = self.text_splitter.split_text(doc["text"])
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": chunk_idx
                        }
                    })
        
        print(f"Created {len(all_chunks)} chunks from {len(pdf_files)} volumes")
        return all_chunks


def load_documents(pdf_dir: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to load and chunk documents
    
    Args:
        pdf_dir: Optional directory path containing PDFs
        
    Returns:
        List of document chunks with metadata
    """
    processor = DocumentProcessor(pdf_dir=pdf_dir)
    return processor.load_and_chunk_documents()


if __name__ == "__main__":
    # Test the document processor
    chunks = load_documents()
    print(f"\nSample chunk:")
    print(f"Text: {chunks[0]['text'][:200]}...")
    print(f"Metadata: {chunks[0]['metadata']}")
