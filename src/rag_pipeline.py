"""
RAG Pipeline for Ponniyin Selvan Chatbot
Handles retrieval, language detection, and response generation
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langdetect import detect
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError as e:
    print(f"Missing dependencies: {e}")
    raise

from .vector_store import VectorStore


class RAGPipeline:
    """RAG Pipeline with bilingual support for Ponniyin Selvan"""
    
    SYSTEM_PROMPT_EN = """You are an expert on Ponniyin Selvan (பொன்னியின் செல்வன்), the epic Tamil historical novel by Kalki Krishnamurthy. 
You have deep knowledge of:
- The Chola dynasty and 10th century Tamil Nadu history
- All major characters: Vandiyathevan, Arulmozhi Varman (Ponniyin Selvan), Kundavai, Nandini, Aditya Karikalan, etc.
- The plot across all 5 volumes: New Floods, The Storm, The Sword, The Crown, and Epitome of Sacrifice

Use the provided context from the novel to answer questions accurately. If the context doesn't contain the answer, use your knowledge but indicate it.
Always be engaging and bring the story to life with vivid descriptions.

Context from the novel:
{context}
"""

    SYSTEM_PROMPT_TA = """நீங்கள் கல்கி கிருஷ்ணமூர்த்தியின் பொன்னியின் செல்வன் என்ற புகழ்பெற்ற தமிழ் வரலாற்று நாவலின் நிபுணர். 
உங்களுக்கு இவை பற்றிய ஆழமான அறிவு உள்ளது:
- சோழ வம்சமும் 10ஆம் நூற்றாண்டு தமிழ்நாட்டின் வரலாறும்
- முக்கிய கதாபாத்திரங்கள்: வந்தியத்தேவன், அருள்மொழி வர்மன், குந்தவை, நந்தினி, ஆதித்ய கரிகாலன் போன்றவர்கள்
- 5 பாகங்களின் கதை: புது வெள்ளம், சுழற்காற்று, கொலை வாள், மணிமகுடம், தியாக சிகரம்

கொடுக்கப்பட்ட சூழலை பயன்படுத்தி கேள்விகளுக்கு துல்லியமாக பதிலளிக்கவும். சூழலில் பதில் இல்லையெனில், உங்கள் அறிவை பயன்படுத்துங்கள்.

நாவலிலிருந்து சூழல்:
{context}
"""

    def __init__(self, model_name: str = "mistral-large-latest"):
        """
        Initialize the RAG pipeline
        
        Args:
            model_name: Mistral model to use
        """
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the Mistral LLM"""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.llm = ChatMistralAI(
            model=self.model_name,
            mistral_api_key=api_key,
            temperature=0.7,
            max_tokens=2048  # Increased for complete responses
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if text is Tamil or English
        
        Args:
            text: Input text
            
        Returns:
            'ta' for Tamil, 'en' for English
        """
        try:
            lang = detect(text)
            return 'ta' if lang == 'ta' else 'en'
        except:
            return 'en'  # Default to English
    
    def _translate_to_english(self, text: str) -> str:
        """
        Translate Tamil text to English for better retrieval
        
        Args:
            text: Tamil text to translate
            
        Returns:
            English translation
        """
        try:
            messages = [
                SystemMessage(content="You are a translator. Translate the following Tamil text to English. Only output the translation, nothing else."),
                HumanMessage(content=text)
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved results into context string"""
        context_parts = []
        for i, result in enumerate(results, 1):
            volume = result['metadata'].get('volume', 'Unknown')
            page = result['metadata'].get('page', '?')
            text = result['text']
            context_parts.append(f"[{volume}, Page {page}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_sources(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract source references from results"""
        sources = []
        for r in results:
            volume = r['metadata'].get('volume', 'Unknown')
            page = r['metadata'].get('page', '?')
            sources.append(f"{volume} (p.{page})")
        return list(set(sources))  # Remove duplicates
    
    def query(
        self, 
        question: str, 
        top_k: int = 5
    ) -> Tuple[str, str, List[str]]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            
        Returns:
            Tuple of (response, detected_language, sources)
        """
        # Detect language
        lang = self._detect_language(question)
        
        # Translate Tamil queries to English for better retrieval
        search_query = question
        if lang == 'ta':
            print(f"Translating Tamil query: {question[:50]}...")
            search_query = self._translate_to_english(question)
            print(f"Translated to: {search_query[:50]}...")
        
        # Retrieve relevant context using English query
        try:
            results = self.vector_store.search(search_query, top_k=top_k)
        except ValueError:
            # Index not built yet
            return (
                "The knowledge base hasn't been built yet. Please run the indexing first.",
                lang,
                []
            )
        
        # Format context
        context = self._format_context(results)
        sources = self._get_sources(results)
        
        # Select prompt based on language
        system_prompt = self.SYSTEM_PROMPT_TA if lang == 'ta' else self.SYSTEM_PROMPT_EN
        system_prompt = system_prompt.format(context=context)
        
        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return answer, lang, sources


def build_index():
    """Build the vector index from PDF documents"""
    from .document_processor import load_documents
    from .vector_store import create_vector_store
    
    print("Loading and processing PDFs...")
    documents = load_documents()
    
    print("Building vector index...")
    vs = create_vector_store(documents, save=True)
    
    print("Index built successfully!")
    return vs


if __name__ == "__main__":
    # Test the RAG pipeline
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        rag = RAGPipeline()
        
        # Test English query
        response, lang, sources = rag.query("Who is Vandiyathevan?")
        print(f"Language: {lang}")
        print(f"Response: {response}")
        print(f"Sources: {sources}")
        
        # Test Tamil query
        response, lang, sources = rag.query("அருள்மொழி வர்மன் யார்?")
        print(f"\nLanguage: {lang}")
        print(f"Response: {response}")
        print(f"Sources: {sources}")
