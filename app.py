import streamlit as st
from chat import OptimizedHybridRetriever, OllamaGenerator, GemmaGenerator, clear_gpu_memory

# ==== AYAR ====
# Eğer güçlü bir bilgisayara geçerseniz burayı False yapın:
USE_OLLAMA_FOR_TESTING = True
# =============

st.set_page_config(page_title="Hukuk RAG Sistemi", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_models():
    retriever = OptimizedHybridRetriever(is_testing=USE_OLLAMA_FOR_TESTING)
    if USE_OLLAMA_FOR_TESTING:
        generator = OllamaGenerator()
    else:
        generator = GemmaGenerator(use_adapter=True)
    return retriever, generator

def main():
    st.title("⚖️ Hukuk RAG Sistemi")
    st.markdown("docs/ klasöründeki belgelere göre hukuk ile ilgili sorularınızı yanıtlayan RAG asistanı.")
    
    with st.sidebar:
        st.header("📄 Belge Yükle")
        uploaded_files = st.file_uploader("Kendi belgelerinizi yükleyin (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Belgeleri Sisteme Ekle"):
            if uploaded_files:
                import os
                from pathlib import Path
                
                # Modellerin yüklendiğinden emin ol
                retriever, _ = load_models()
                
                with st.status("🚀 Belgeler işleniyor ve sisteme ekleniyor... (GPU geçici olarak kullanılacak)", expanded=True) as status:
                    os.makedirs("docs", exist_ok=True)
                    saved_paths = []
                    for uf in uploaded_files:
                        file_path = os.path.join("docs", uf.name)
                        with open(file_path, "wb") as f:
                            f.write(uf.getbuffer())
                        saved_paths.append(Path(file_path))
                        
                    try:
                        num_chunks = retriever.add_documents(saved_paths)
                        if num_chunks > 0:
                            status.update(label=f"✅ Başarılı! {len(saved_paths)} belge işlendi ve {num_chunks} yeni parça eklendi. GPU boşaltıldı.", state="complete", expanded=False)
                        else:
                            status.update(label="Bilgi: Bu belgeler zaten tamamen veritabanında mevcut.", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label=f"❌ Hata: {str(e)}", state="error", expanded=True)
            else:
                st.warning("Lütfen önce bir belge seçin.")
                
        st.markdown("---")
        
        st.header("⚙️ Ayarlar")
        pool = st.slider("Reranker Pool (Havuz) Boyutu", min_value=5, max_value=200, value=20, step=5,
                         help="Retrieval aşamasında ilk getirilecek ve daha sonra reranker modelinden geçirilecek belge havuzunun boyutunu belirler.")
        top_k = st.slider("Cevaba Verilecek Kaynak Sayısı (Top K)", min_value=1, max_value=10, value=5, step=1,
                          help="Modele bağlam olarak sunulacak nihai belge sayısını belirler.")
        st.markdown("---")
        st.info("Modeller ilk çalıştırmada belleğe yüklenir, bu nedenle ilk soru biraz zaman alabilir.")

    # Modellerin yüklenmesi
    with st.spinner("Modeller yükleniyor / kontrol ediliyor... (Bu işlem ilk açılışta zaman alabilir)"):
        retriever, generator = load_models()
        
    # Sohbet arayüzü durumu
    if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
        st.session_state.messages = []
        
    # Eğer messages listesi içindeki öğeler dict değilse (örn. eski bir oturumdan string kalmışsa) temizle
    if any(not isinstance(msg, dict) for msg in st.session_state.messages):
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg.get("role", "unknown")):
            st.markdown(msg.get("content", ""))
            
    if prompt := st.chat_input("Hukuki sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Belgeler aranıyor ve yanıt üretiliyor..."):
                try:
                    chunks = retriever.search(prompt, top_k=top_k, pool=pool)
                    if chunks:
                        # Streaming ile yanıt göster
                        response_placeholder = st.empty()
                        full_response = ""
                        stream = generator.generate_stream(prompt, chunks)
                        
                        for chunk in stream:
                            if 'message' in chunk and 'content' in chunk['message']:
                                full_response += chunk['message']['content']
                                response_placeholder.markdown(full_response + "▌")
                                
                        response_placeholder.markdown(full_response)
                        
                        with st.expander("📚 Kaynaklar & Alakalı Metinler"):
                            for i, c in enumerate(chunks, 1):
                                st.markdown(f"**[{i}] Belge:** {c['doc_id']} | **İlgi Skoru (Rerank):** {c['rerank_score']:.3f}")
                                st.caption(c['text'][:500] + "..." if len(c['text']) > 500 else c['text'])
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        clear_gpu_memory()  # İşlem sonu güvenliği
                    else:
                        st.warning("Eşleşen belge bulunamadı. Lütfen docs/ klasörüne belge ekleyip ingest işlemini çalıştırın.")
                        st.session_state.messages.append({"role": "assistant", "content": "Eşleşen belge bulunamadı."})
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
