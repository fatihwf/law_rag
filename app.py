import streamlit as st
from chat import OptimizedHybridRetriever, GemmaGenerator

st.set_page_config(page_title="Hukuk RAG Sistemi", page_icon="⚖️", layout="wide")

@st.cache_resource
def load_models():
    retriever = OptimizedHybridRetriever()
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
                
                with st.spinner("Belgeler işleniyor ve sisteme ekleniyor... (Biraz sürebilir)"):
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
                            st.success(f"Başarılı! {len(saved_paths)} belge işlendi ve {num_chunks} yeni parça veritabanına eklendi.")
                        else:
                            st.info("Bu belgeler zaten tamamen veritabanında mevcut.")
                    except Exception as e:
                        st.error(f"Belgeler eklenirken hata oluştu: {str(e)}")
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Hukuki sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Belgeler aranıyor ve yanıt üretiliyor..."):
                try:
                    chunks = retriever.search(prompt, top_k=top_k, pool=pool)
                    if chunks:
                        answer = generator.generate(prompt, chunks)
                        
                        # Sonuçları göster
                        st.markdown(answer)
                        
                        with st.expander("📚 Kaynaklar & Alakalı Metinler"):
                            for i, c in enumerate(chunks, 1):
                                st.markdown(f"**[{i}] Belge:** {c['doc_id']} | **İlgi Skoru (Rerank):** {c['rerank_score']:.3f}")
                                st.caption(c['text'][:500] + "..." if len(c['text']) > 500 else c['text'])
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.warning("Eşleşen belge bulunamadı. Lütfen docs/ klasörüne belge ekleyip ingest işlemini çalıştırın.")
                        st.session_state.messages.append({"role": "assistant", "content": "Eşleşen belge bulunamadı."})
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
