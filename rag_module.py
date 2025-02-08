import re
import hashlib
import fitz
import numpy as np
import faiss
import ollama
import langdetect
import nltk
from pathlib import Path
from typing import List, Tuple

nltk.download("punkt", quiet=True)

def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'\n{3,}', '\n\n', re.sub(r'[^\w\s\-.,;:?!]', '', text))).strip()

def semantic_chunker(text, max_length=512, overlap=64):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_length + sentence_word_count > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = []
            while current_chunk and len(" ".join(overlap_words + [current_chunk[-1]]).split()) < overlap:
                overlap_words.insert(0, current_chunk.pop())
            current_chunk = overlap_words.copy()
            current_length = sum(len(s.split()) for s in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def load_pdf_document(file_path):
    try:
        with fitz.open(file_path) as doc:
            full_text = ""
            for page in doc:
                full_text += clean_text(page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)) + "\n\n"
        return semantic_chunker(full_text)
    except Exception as e:
        raise ValueError(f"PDF processing error: {str(e)}")

def generate_index_name(pdf_path):
    file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
    return Path(f"index_{file_hash}.faiss"), Path(f"docs_{file_hash}.npy")

def build_faiss_index(documents, embedder):
    embeddings = embedder.encode(documents, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 200
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def load_faiss_index(embedder, pdf_file_path):
    pdf_path = Path(pdf_file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file_path}")
    
    index_file, docs_file = generate_index_name(pdf_file_path)
    
    try:
        if index_file.exists() and docs_file.exists():
            return faiss.read_index(str(index_file)), np.load(docs_file, allow_pickle=True).tolist()
    except Exception:
        pass
    
    documents = load_pdf_document(pdf_file_path)
    index, _ = build_faiss_index(documents, embedder)
    faiss.write_index(index, str(index_file))
    np.save(str(docs_file), np.array(documents))
    return index, documents

def hybrid_search(query, embedder, index, documents, top_k=3):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    _, semantic_indices = index.search(query_embedding, top_k * 2)
    
    query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))
    keyword_scores = []
    for i, doc in enumerate(documents):
        keyword_scores.append((i, sum(1 for word in query_keywords if word in doc.lower())))
    
    top_keywords = [i for i, _ in sorted(keyword_scores, key=lambda x: x[1], reverse=True)[:top_k]]
    combined_indices = set(semantic_indices[0].tolist() + top_keywords)
    
    results = []
    for idx in combined_indices:
        if len(results) >= top_k:
            break
        results.append(documents[idx])
    return results

def query_ollama(query, embedder, index, documents, top_k=3, temperature=0.7):
    context_chunks = hybrid_search(query, embedder, index, documents, top_k)
    
    try:
        lang = langdetect.detect(query)
    except:
        lang = "en"

    system_prompt = f"""You are an expert research assistant. Analyze the context and respond in the same language as the question.
Respond in: {"French" if lang == "fr" else "English"}
Include:
1. Clear answer
2. Supporting details
3. Logical reasoning"""

    user_prompt = f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion: {query}\nAnswer with page references when possible."

    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": temperature, "num_ctx": 4096, "repeat_penalty": 1.2}
        )
        return response['message']['content'], context_chunks
    except Exception as e:
        return f"Error generating response: {str(e)}", []