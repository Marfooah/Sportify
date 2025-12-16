import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set. Set it in environment variables before running.")

client = Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
LLM_MODEL = "llama-3.3-70b-versatile"

DATA_FILES = [
    "data/overview.txt",
    "data/sports.txt",
    "data/timings.txt",
    "data/memberships.txt",
    "data/booking_rules.txt",
    "data/safety.txt",
    "data/coaches.txt",
    "data/members.txt",
    "data/faqs.txt"
]

# ---------------- DATA INGESTION ----------------
def load_documents():
    docs = []
    for file in DATA_FILES:
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

docs = load_documents()
chunks = []
for d in docs:
    chunks.extend(chunk_text(d))

embeddings = EMBED_MODEL.encode(chunks)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ---------------- RAG QUERY ----------------
def rag_answer(query, history):
    q_embed = EMBED_MODEL.encode([query])
    _, idx = index.search(np.array(q_embed), k=5)
    context = "\n".join([chunks[i] for i in idx[0]])

    messages = [
        {"role": "system", "content": "You are Sportify, a professional AI assistant for a sports complex. Answer clearly and professionally."}
    ]

    for h in history:
        messages.append(h)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    })

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.set_page_config(page_title="Sportify AI", layout="centered")
st.title("🏟️ Sportify – Sports Complex AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask about sports, timings, memberships, bookings...")



if st.button("Ask") and user_input:
    answer = rag_answer(user_input, st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Sportify:** {msg['content']}")
