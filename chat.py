from __future__ import annotations

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import argparse
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
for _lib in [
    "httpx", "httpcore", "huggingface_hub", "sentence_transformers",
    "transformers", "torch", "chromadb", "urllib3", "filelock", "accelerate",
]:
    logging.getLogger(_lib).setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Yapılandırma (Paths are now relative to the 'final' folder)
# ---------------------------------------------------------------------------
CHROMA_DB_PATH      = Path("chroma_db")
COLLECTION_NAME     = "legal_docs"
ADAPTER_PATH        = Path("models/finetuned_gemma4-e2b")

EMBED_MODEL_NAME    = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEVICE              = "cuda"

# OPTIMIZED PARAMETERS
DEFAULT_TOP_K       = 5
DEFAULT_POOL        = 20
ALPHA_WEIGHT        = 0.7  

MAX_NEW_TOKENS      = 1024
CONTEXT_MAX_CHARS   = 48000

class OptimizedHybridRetriever:
    def __init__(self):
        import torch
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self._device = DEVICE if torch.cuda.is_available() else "cpu"
        log.info(f"OptimizedHybridRetriever — device: {self._device}")

        log.info("ChromaDB yükleniyor...")
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)
        log.info(f"  {self.collection.count()} kayıt yüklendi.")

        log.info("BM25 indeksi kuruluyor...")
        self._load_bm25()

        log.info(f"Embedder yükleniyor: {EMBED_MODEL_NAME}")
        self.embedder = SentenceTransformer(
            EMBED_MODEL_NAME,
            device=self._device,
            local_files_only=False, # Set to false if missing local weights
        )

        log.info(f"Reranker yükleniyor: {RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=8192,
            trust_remote_code=True,
            device=self._device,
            local_files_only=False, # Set to false if missing local weights
        )
        log.info("OptimizedHybridRetriever hazır [OK]")

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE).split()

    def _load_bm25(self):
        from rank_bm25 import BM25Okapi
        total = self.collection.count()
        all_ids, all_docs, all_metas = [], [], []
        for offset in range(0, total, 5000):
            res = self.collection.get(
                limit=5000, offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids.extend(res["ids"])
            all_docs.extend(res["documents"])
            all_metas.extend(res["metadatas"])
        
        if not all_docs:
            log.warning("Veritabaninda belge yok! Lutfen docs/ klasorune dosya koyup ingest.py calistirin.")
            all_docs = ["dummy text"] # Prevent crash if empty
            
        self._all_ids   = all_ids
        self._all_docs  = all_docs
        self._all_metas = all_metas
        self._bm25      = BM25Okapi([self._tokenize(d) for d in all_docs])
        log.info(f"  BM25 — {len(all_docs)} chunk.")

    def _dense_search(self, query: str, top_n: int) -> list[dict]:
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True, device=self._device,
            show_progress_bar=False
        ).tolist()
        res = self.collection.query(
            query_embeddings=q_emb, n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )
        if not res["ids"][0]: return []
        
        return [{
            "chunk_id":    cid,
            "text":        res["documents"][0][i],
            "metadata":    res["metadatas"][0][i],
            "dense_score": 1.0 - res["distances"][0][i],
        } for i, cid in enumerate(res["ids"][0])]

    def _bm25_search(self, query: str, top_n: int) -> list[dict]:
        import numpy as np
        scores  = self._bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [{
            "chunk_id":   self._all_ids[i],
            "text":       self._all_docs[i],
            "metadata":   self._all_metas[i] if self._all_metas else {},
            "bm25_score": float(scores[i]),
        } for i in top_idx]

    @staticmethod
    def _alpha_fusion(dense_res: list[dict], sparse_res: list[dict], alpha: float) -> list[dict]:
        dense_scores = {x["chunk_id"]: x["dense_score"] for x in dense_res}
        sparse_scores = {x["chunk_id"]: x["bm25_score"] for x in sparse_res}
        
        all_chunks = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        if not all_chunks:
            return []
            
        d_min, d_max = min(dense_scores.values()) if dense_scores else 0, max(dense_scores.values()) if dense_scores else 1
        s_min, s_max = min(sparse_scores.values()) if sparse_scores else 0, max(sparse_scores.values()) if sparse_scores else 1
        
        def norm(val, v_min, v_max):
            if v_max == v_min: return 0.0
            return (val - v_min) / (v_max - v_min)

        fused = []
        data_map = {r["chunk_id"]: r for r in dense_res}
        for r in sparse_res:
            if r["chunk_id"] not in data_map:
                data_map[r["chunk_id"]] = r

        for cid in all_chunks:
            d_norm = norm(dense_scores.get(cid, d_min), d_min, d_max) if cid in dense_scores else 0.0
            s_norm = norm(sparse_scores.get(cid, s_min), s_min, s_max) if cid in sparse_scores else 0.0
            
            final_score = (alpha * d_norm) + ((1.0 - alpha) * s_norm)
            
            c_data = data_map[cid].copy()
            c_data["fusion_score"] = final_score
            fused.append(c_data)
            
        fused.sort(key=lambda x: x["fusion_score"], reverse=True)
        return fused

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, pool: int = DEFAULT_POOL, rerank: bool = True) -> list[dict]:
        search_pool = max(75, pool) 
        fused = self._alpha_fusion(
            self._dense_search(query, search_pool),
            self._bm25_search(query, search_pool),
            alpha=ALPHA_WEIGHT
        )[:pool]

        if rerank and fused:
            scores = self.reranker.predict([(query, c["text"]) for c in fused])
            for h, sc in zip(fused, scores):
                h["rerank_score"] = float(sc)
            fused.sort(key=lambda x: x["rerank_score"], reverse=True)

        return [{
            "chunk_id":     h["chunk_id"],
            "doc_id":       h.get("metadata", {}).get("doc_id", "Unknown"),
            "text":         h["text"],
            "fusion_score": round(h.get("fusion_score", 0.0), 6),
            "rerank_score": round(h.get("rerank_score", 0.0), 4) if rerank else None,
        } for h in fused[:top_k]]


class GemmaGenerator:
    SYSTEM_PROMPT = (
        "Sen bir uzman asistansın. Sana verilen belge parçalarını "
        "dikkatlice inceleyerek SADECE sorulan soruyu yanıtla. "
        "Kesinlikle 'Verilen metin şununla ilgilidir' gibi özetler yapma, doğrudan cevaba gir. "
        "Yalnızca verilen bağlama dayanarak kısa ve net cevap ver."
    )

    def __init__(self, use_adapter: bool = False):
        from unsloth import FastModel
        import json

        if use_adapter and ADAPTER_PATH.exists():
            log.info("Gemma-4-E2B + LoRA yükleniyor...")
            model_name_to_load = str(ADAPTER_PATH.resolve())
        else:
            log.info("Gemma-4-E2B Base Model yükleniyor (Adapter bulunamadi)...")
            model_name_to_load = "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit"

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name_to_load,
            max_seq_length=16384,
            load_in_4bit=True,
            dtype=None,
        )
        FastModel.for_inference(self.model)
        log.info("GemmaGenerator hazir [OK]")

    def generate(self, question: str, context_chunks: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        import torch
        
        ctx_texts = []
        total_len = 0
        for c in context_chunks:
            txt = f"[Belge: {c['doc_id']}]\n{c['text']}"
            if total_len + len(txt) > CONTEXT_MAX_CHARS: break
            ctx_texts.append(txt)
            total_len += len(txt)

        context_str = "\n\n---\n\n".join(ctx_texts)
        user_prompt = f"Bağlam:\n{context_str}\n\nSoru: {question}"

        messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{self.SYSTEM_PROMPT}\n\n{user_prompt}"
            }
        ]
    }
]

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
            )

        gen_tokens = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def interactive_loop():
    print("="*60)
    print("RAG SISTEMI (HUKUK)")
    print("Veritabani: docs/ klasorundeki belgeler")
    print("="*60)
    
    retriever = OptimizedHybridRetriever()
    generator = GemmaGenerator(use_adapter=True)
    
    print("\nSistem Hazir! Soru sorabilirsiniz (Cikmak icin 'q' veya 'exit').")
    
    while True:
        try:
            q = input("\n> Soru: ").strip()
            if q.lower() in ["q", "exit", "quit"]: break
            if not q: continue
                
            print("  Belgeler taranıyor...")
            chunks = retriever.search(q)
            
            if not chunks:
                print("  Eslenen belge bulunamadi. Lutfen docs/ klasorune belge ekleyip ingest islemini calistirin.")
                continue
                
            print(f"  {len(chunks)} alakali metin bulundu. Yanit uretiliyor...\n")
            answer = generator.generate(q, chunks)
            
            print("="*60)
            print("YANIT:")
            print(answer)
            print("="*60)
            print("KAYNAKLAR:")
            for i, c in enumerate(chunks, 1):
                print(f" [{i}] Belge: {c['doc_id']} | Ilgi Skoru: {c['rerank_score']:.3f}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Hata: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Soruyu arguman olarak ver")
    args = parser.parse_args()

    if args.query:
        retriever = OptimizedHybridRetriever()
        generator = GemmaGenerator(use_adapter=True)
        chunks = retriever.search(args.query)
        if chunks:
            ans = generator.generate(args.query, chunks)
            print("\nYANIT:\n", ans)
        else:
            print("Veritabaninda belge yok.")
    else:
        interactive_loop()
