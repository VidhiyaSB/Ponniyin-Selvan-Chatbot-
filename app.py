"""
Ponniyin Selvan RAG Chatbot - Main Gradio Application
A bilingual chatbot for exploring Kalki's epic historical novel
"""
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from gradio import ChatMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.rag_pipeline import RAGPipeline, build_index
from src.voice_handler import VoiceHandler
from src.vector_store import VectorStore


# Initialize components
rag_pipeline = None
voice_handler = None


def initialize_components():
    """Initialize RAG pipeline and voice handler"""
    global rag_pipeline, voice_handler
    
    try:
        rag_pipeline = RAGPipeline()
        print("RAG Pipeline initialized")
    except Exception as e:
        print(f"Error initializing RAG Pipeline: {e}")
        rag_pipeline = None
    
    try:
        voice_handler = VoiceHandler()
        print(f"Voice Handler initialized: {voice_handler.is_available()}")
    except Exception as e:
        print(f"Error initializing Voice Handler: {e}")
        voice_handler = None


def check_index_exists():
    """Check if the vector index has been built"""
    vs = VectorStore()
    return vs.load_index()


def chat_response(message: str, history: list, enable_voice: bool):
    """
    Process chat message and generate response
    
    Args:
        message: User's message
        history: Chat history
        enable_voice: Whether to generate voice output
        
    Returns:
        Tuple of (response text, audio file path or None, sources text)
    """
    global rag_pipeline, voice_handler
    
    if rag_pipeline is None:
        return (
            "⚠️ RAG Pipeline not initialized. Please check your API keys.",
            None,
            "No sources available"
        )
    
    # Check if index exists
    if not check_index_exists():
        return (
            "📚 The knowledge base hasn't been built yet. Click 'Build Index' to process the PDFs first.",
            None,
            "No sources available"
        )
    
    try:
        # Query RAG pipeline
        response, lang, sources = rag_pipeline.query(message)
        
        # Format sources as HTML with HIGH CONTRAST inline styles for HF Spaces
        if sources:
            sources_items = "".join([f"<div style='margin: 5px 0; color: #FFFFFF !important;'>• {s}</div>" for s in sources[:3]])
            sources_text = f"<div style='background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-radius: 12px; padding: 18px;'>📖 <strong style='color: #FFD700 !important;'>Sources:</strong>{sources_items}</div>"
        else:
            sources_text = "<div style='background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-radius: 12px; padding: 18px; color: #FFFFFF !important;'>No specific sources found</div>"
        
        # Generate voice if enabled
        audio_path = None
        if enable_voice and voice_handler and voice_handler.is_available():
            try:
                audio_bytes = voice_handler.text_to_speech(response, language=lang)
                if audio_bytes:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        f.write(audio_bytes)
                        audio_path = f.name
            except Exception as e:
                print(f"Voice generation error: {e}")
        
        return response, audio_path, sources_text
        
    except Exception as e:
        return (
            f"❌ Error processing your question: {str(e)}",
            None,
            "Error occurred"
        )


def create_interface():
    """Create the Gradio interface"""
    
    # Load custom CSS
    css_path = Path(__file__).parent / "assets" / "style.css"
    custom_css = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
    
    # Gradio 6.x: css moved to head parameter
    with gr.Blocks(title="Ponniyin Selvan Chatbot", head=f"<style>{custom_css}</style>") as demo:

        # Header
        gr.HTML("""
            <div class="app-header">
                <h1 class="app-title">🏛️ பொன்னியின் செல்வன் 🏛️</h1>
                <h1 class="app-title" style="font-size: 1.8rem;">Ponniyin Selvan</h1>
                <p class="app-subtitle">
                    Explore Kalki's Masterpiece | 
                    கல்கியின் காவியத்தைப் பற்றி ஆராயுங்கள்
                </p>
            </div>
        """)
        
        with gr.Row():
            # Main chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    elem_classes=["chatbot-container"]
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about Vandiyathevan, Kundavai, the Chola dynasty... (Tamil or English)",
                        label="Your Question",
                        scale=4,
                        elem_classes=["input-container"]
                    )
                    submit_btn = gr.Button("Send 📨", variant="primary", scale=1)
                
                with gr.Row():
                    voice_toggle = gr.Checkbox(
                        label="🔊 Enable Voice Response",
                        value=False,
                        elem_classes=["voice-toggle"]
                    )
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", elem_id="clear-btn")
            
            # Side panel
            with gr.Column(scale=1):
                # Audio output
                audio_output = gr.Audio(
                    label="🎧 Voice Response",
                    type="filepath",
                    autoplay=True,
                    visible=True
                )
                
                # Sources panel - HIGH CONTRAST for HF Spaces
                sources_display = gr.HTML(
                    value="<div style='background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-radius: 12px; padding: 18px;'>📖 <strong style='color: #FFD700 !important;'>Sources will appear here</strong></div>",
                    elem_classes=["sources-panel"]
                )
                
                gr.HTML("<div class='decorative-border'></div>")
                
                gr.HTML("<div class='decorative-border'></div>")
                
                # Character quick info - HIGH CONTRAST for HF Spaces
                gr.HTML("""
                    <div style="background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-left: 5px solid #FFD700 !important; border-radius: 12px; padding: 16px 18px; margin: 12px 0;">
                        <div style="color: #FFD700 !important; font-size: 1.15rem; font-weight: bold; margin-bottom: 6px;">🗡️ வந்தியத்தேவன்</div>
                        <div style="color: #FFFFFF !important; font-size: 0.9rem; line-height: 1.4;">Vallavaraiyan Vandiyathevan - The brave Vanar warrior</div>
                    </div>
                    <div style="background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-left: 5px solid #FFD700 !important; border-radius: 12px; padding: 16px 18px; margin: 12px 0;">
                        <div style="color: #FFD700 !important; font-size: 1.15rem; font-weight: bold; margin-bottom: 6px;">👑 அருள்மொழி வர்மன்</div>
                        <div style="color: #FFFFFF !important; font-size: 0.9rem; line-height: 1.4;">Ponniyin Selvan - The beloved of Ponni (Kaveri)</div>
                    </div>
                    <div style="background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-left: 5px solid #FFD700 !important; border-radius: 12px; padding: 16px 18px; margin: 12px 0;">
                        <div style="color: #FFD700 !important; font-size: 1.15rem; font-weight: bold; margin-bottom: 6px;">👸 குந்தவை</div>
                        <div style="color: #FFFFFF !important; font-size: 0.9rem; line-height: 1.4;">Princess Kundavai - The wise Chola princess</div>
                    </div>
                """)
                
                gr.HTML("<div class='decorative-border'></div>")
                
                # Optional API key settings (hidden by default)
                with gr.Accordion("⚙️ Settings (Optional API Keys)", open=False):
                    gr.HTML("<p style='color: white !important; font-weight: bold; margin-bottom: 10px;'>Use your own API keys if the default ones run out</p>")
                    user_mistral_key = gr.Textbox(
                        label="Mistral API Key",
                        placeholder="Enter your Mistral API key (optional)",
                        type="password",
                        lines=1
                    )
                    user_elevenlabs_key = gr.Textbox(
                        label="ElevenLabs API Key", 
                        placeholder="Enter your ElevenLabs API key (optional)",
                        type="password",
                        lines=1
                    )
                    apply_keys_btn = gr.Button("Apply Keys", variant="secondary", size="sm")
        
        # Example questions
        gr.Examples(
            examples=[
                ["Who is Vandiyathevan and what is his role in the story?"],
                ["Tell me about Kundavai and her relationship with Vandiyathevan"],
                ["What happens at the end of Ponniyin Selvan?"],
                ["வந்தியத்தேவன் யார்?"],
                ["நந்தினி பற்றி சொல்லுங்கள்"],
                ["சோழ வம்சத்தின் வரலாறு என்ன?"]
            ],
            inputs=msg,
            label="💡 Try these questions"
        )
        
        # Event handlers
        def respond(message, history, enable_voice):
            if not message.strip():
                return history, None, "Please enter a question"
            
            response, audio, sources = chat_response(message, history, enable_voice)
            # Use Gradio 6.x ChatMessage format
            history = history + [
                ChatMessage(role="user", content=message),
                ChatMessage(role="assistant", content=response)
            ]
            return history, audio, sources
        
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, voice_toggle],
            outputs=[chatbot, audio_output, sources_display]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, voice_toggle],
            outputs=[chatbot, audio_output, sources_display]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            lambda: ([], None, "<div style='background: #1a1a2e !important; border: 2px solid #FFD700 !important; border-radius: 12px; padding: 18px;'>📖 <strong style='color: #FFD700 !important;'>Sources will appear here</strong></div>"),
            outputs=[chatbot, audio_output, sources_display]
        )
        
        # Apply custom API keys
        def apply_custom_keys(mistral_key, elevenlabs_key):
            global rag_pipeline, voice_handler
            import os
            
            status_msgs = []
            
            # Apply Mistral key if provided
            if mistral_key and mistral_key.strip():
                os.environ["MISTRAL_API_KEY"] = mistral_key.strip()
                try:
                    rag_pipeline = RAGPipeline()
                    status_msgs.append("✅ Mistral API key applied")
                except Exception as e:
                    status_msgs.append(f"❌ Mistral error: {str(e)}")
            
            # Apply ElevenLabs key if provided  
            if elevenlabs_key and elevenlabs_key.strip():
                os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key.strip()
                try:
                    voice_handler = VoiceHandler()
                    status_msgs.append("✅ ElevenLabs API key applied")
                except Exception as e:
                    status_msgs.append(f"❌ ElevenLabs error: {str(e)}")
            
            if not status_msgs:
                return "ℹ️ No keys provided"
            return " | ".join(status_msgs)
        
        apply_keys_btn.click(
            apply_custom_keys,
            inputs=[user_mistral_key, user_elevenlabs_key],
            outputs=[gr.Textbox(visible=False)]  # Silent output
        )
        
        # Copyright Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 20px; padding: 10px; color: #D4A84B; opacity: 0.8; font-size: 0.9rem;">
                Made with ❤️ by Vidhiya S B
            </div>
        """)
    
    return demo


# Initialize on module load
initialize_components()


if __name__ == "__main__":
    demo = create_interface()
    # For HuggingFace Spaces: don't specify server_name/port (HF manages this)
    # For local: these will be ignored if HF environment variables are set
    demo.launch()
