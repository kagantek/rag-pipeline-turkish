import os
import tempfile
from typing import List, Dict, Any, Tuple
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODELS,
    DEFAULT_LLM_MODEL,
    GROQ_API_KEY,
    UI_TEXTS
)
from embeddings import get_embedder
from vector_store import get_vector_store
from retrieval import get_retrieval_engine
from llm_generator import get_llm_generator


# Sayfa düzeni
st.set_page_config(
    page_title="Atlas ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "embedder_loaded" not in st.session_state:
        st.session_state.embedder_loaded = False
    if "sources" not in st.session_state:
        st.session_state.sources = []


init_session_state()


def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    pages_data = []
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text(layout=True)
                if text and text.strip():
                    pages_data.append({
                        "text": text,
                        "page_number": page_num
                    })
    except Exception as e:
        st.error(f"PDF okuma hatası: {e}")
    finally:
        os.unlink(tmp_path)
    
    return pages_data


def chunk_text(text: str, source: str, page_number: int) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    return [
        {
            "text": chunk,
            "source": source,
            "chunk_index": i,
            "page_number": page_number
        }
        for i, chunk in enumerate(chunks)
    ]


def process_uploaded_files(uploaded_files) -> int:
    vector_store = get_vector_store()
    
    total_chunks = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pdf_file in enumerate(uploaded_files):
        status_text.text(f"İşleniyor: {pdf_file.name}")
        
        try:
            pages_data = extract_text_from_pdf(pdf_file)
            
            if not pages_data:
                st.warning(f"{pdf_file.name} dosyasından metin çıkarılamadı.")
                continue
            
            all_texts = []
            all_metadatas = []
            
            for page_data in pages_data:
                chunks = chunk_text(
                    page_data["text"], 
                    pdf_file.name, 
                    page_data["page_number"]
                )
                
                for chunk in chunks:
                    all_texts.append(chunk["text"])
                    all_metadatas.append({
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "page_number": chunk["page_number"]
                    })
            
            if all_texts:
                vector_store.add_documents(all_texts, all_metadatas)
                total_chunks += len(all_texts)
                
        except Exception as e:
            st.error(f"{pdf_file.name} işlenirken hata: {e}")
            continue
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return total_chunks


def render_sidebar():
    
    st.sidebar.title("Ayarlar")
    
    if not GROQ_API_KEY:
        st.sidebar.error(UI_TEXTS["error_api_key"])
        st.sidebar.code("GROQ_API_KEY=your_key_here", language="bash")
        st.sidebar.markdown("[Groq Console'dan API Key Alın](https://console.groq.com/keys)")
    
    st.sidebar.divider()
    
    st.sidebar.subheader(UI_TEXTS["model_select"])
    selected_model_name = st.sidebar.selectbox(
        "Model",
        options=list(LLM_MODELS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.selected_model = LLM_MODELS[selected_model_name]
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Arama Modu")
    use_reranking = st.sidebar.toggle(
        UI_TEXTS["rerank_toggle"],
        value=True,
        help=UI_TEXTS["rerank_help"]
    )
    
    st.session_state.use_reranking = use_reranking
    
    if use_reranking:
        st.sidebar.success(f"{UI_TEXTS['search_advanced']}")
    else:
        st.sidebar.info(f"ℹ{UI_TEXTS['search_standard']}")
    
    st.sidebar.divider()
    
    st.sidebar.subheader(UI_TEXTS["upload_label"])
    uploaded_files = st.sidebar.file_uploader(
        "PDF Yükle",
        type=["pdf"],
        accept_multiple_files=True,
        help=UI_TEXTS["upload_help"],
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.sidebar.button(UI_TEXTS["process_button"], type="primary", use_container_width=True):
            with st.spinner(UI_TEXTS["processing"]):
                chunk_count = process_uploaded_files(uploaded_files)
                st.session_state.documents_processed = True
                st.session_state.chunk_count = chunk_count
                st.sidebar.success(UI_TEXTS["processing_complete"].format(count=len(uploaded_files)))
                st.sidebar.info(UI_TEXTS["chunk_count"].format(count=chunk_count))
                st.rerun()
    
    if st.session_state.documents_processed:
        st.sidebar.divider()
        st.sidebar.subheader("Veritabanı Durumu")
        st.sidebar.metric("Toplam Chunk", st.session_state.chunk_count)
        
        if st.sidebar.button("Veritabanını Sıfırla", use_container_width=True):
            try:
                get_vector_store().clear_collection()
                st.session_state.documents_processed = False
                st.session_state.chunk_count = 0
                st.session_state.chat_history = []
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Sıfırlama hatası: {e}")


def render_chat():
    
    st.title(UI_TEXTS["title"])
    st.caption(UI_TEXTS["subtitle"])
    
    if not st.session_state.documents_processed:
        st.info(UI_TEXTS["no_docs"])
        
        with st.expander("Nasıl Kullanılır?"):
            st.markdown("""
            ### Adım 1: API Anahtarı Ayarlayın
            1. [Groq Console](https://console.groq.com/keys) adresinden ücretsiz API anahtarı alın
            2. Proje klasöründe `.env` dosyası oluşturun
            3. `GROQ_API_KEY=your_key_here` satırını ekleyin
            
            ### Adım 2: PDF Yükleyin
            1. Sol panelden PDF dosyalarınızı seçin
            2. "Dökümanları İşle" butonuna tıklayın
            3. İşlem tamamlandığında soru sormaya başlayın
            
            ### Örnek Sorular
            - "KDV oranı nedir?"
            - "Fatura kesim tarihi ne zaman?"
            - "Amortisman süresi kaç yıl?"
            """)
        return
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander(f"{UI_TEXTS['sources_label']} ({len(message['sources'])} kaynak)"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-header">
                                {src['source']} 
                                <span class="score-badge">Skor: {src['score']:.4f}</span>
                            </div>
                            <div>{src['text'][:500]}{'...' if len(src['text']) > 500 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    if prompt := st.chat_input(UI_TEXTS["query_placeholder"]):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(UI_TEXTS["thinking"]):
                response, sources, debug_info = generate_response(prompt)
            
            st.markdown(response)
            
            with st.expander(f"{UI_TEXTS['sources_label']} ({len(sources)} kaynak)"):
                for src in sources:
                    original_score = src.get('original_score', src['score'])
                    st.markdown(f"""
                    <div class="source-box">
                        <div class="source-header">
                            {src['source']} 
                            <span class="score-badge">Skor: {src['score']:.4f}</span>
                            {f'<span style="margin-left:8px; font-size:0.75rem;">(Vector: {original_score:.4f})</span>' if st.session_state.use_reranking else ''}
                        </div>
                        <div>{src['text'][:500]}{'...' if len(src['text']) > 500 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("Arama Detayları (Debug)"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Arama Modu", "Reranking" if debug_info.get("use_reranking") else "Standard")
                col2.metric("Stage 1 Sonuç", debug_info.get("stage1_count", 0))
                col3.metric("Final Sonuç", len(sources))
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })


def generate_response(query: str) -> Tuple[str, List[Dict], Dict]:
    try:
        retrieval_engine = get_retrieval_engine()
        results, debug_info = retrieval_engine.retrieve(
            query=query,
            use_reranking=st.session_state.get("use_reranking", True)
        )
        
        context = retrieval_engine.build_context(results)
        sources = retrieval_engine.format_sources(results)
        
        chat_history = []
        for msg in st.session_state.chat_history[-6:]:
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        llm = get_llm_generator()
        response = llm.generate(
            question=query,
            context=context,
            chat_history=chat_history,
            model_id=st.session_state.get("selected_model", DEFAULT_LLM_MODEL)
        )
        
        return response, sources, debug_info
        
    except Exception as e:
        error_msg = "Bağlantı hatası, lütfen tekrar deneyin."
        st.error(f"{error_msg}")
        print(f"Generate response error: {e}")
        return error_msg, [], {"error": str(e)}


def main():
    """Main application entry point."""
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
