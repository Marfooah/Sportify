# 🏟️ Sportify AI – RAG-Powered Sports Complex Assistant

An intelligent Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, FAISS, Sentence Transformers, and Groq LLMs.

Sportify AI enables users to ask natural language questions about a sports complex and receive accurate, context-aware answers based on the organization's knowledge base.

Instead of relying solely on a Large Language Model, the system first retrieves relevant information from internal documents and then generates responses using an LLM, reducing hallucinations and improving accuracy.

---
🚀 Live Demo: https://sportify37.streamlit.app/

## 📸 Application Preview

### Home Page
<img width="2876" height="866" alt="image" src="https://github.com/user-attachments/assets/833ccc6f-73c9-4bb3-8748-124c5b945c06" />

### Membership Information
<img width="2894" height="1590" alt="image" src="https://github.com/user-attachments/assets/bd78f1e3-5306-4e6d-a3be-d9e867e9498c" />
<img width="2894" height="356" alt="image" src="https://github.com/user-attachments/assets/a4c4a39d-f3b2-4723-a06f-f7815a871e44" />

### Facility Timings
<img width="2894" height="536" alt="image" src="https://github.com/user-attachments/assets/661e5d2c-4da5-4934-b053-a953c6e86979" />

### Coaches Information
<img width="2894" height="546" alt="image" src="https://github.com/user-attachments/assets/1d9f84ab-a4b3-4e05-820a-497c91fce714" />

### Conversational Memory
<img width="2894" height="1556" alt="image" src="https://github.com/user-attachments/assets/818a346f-37cf-4d55-83f9-da705a53f25c" />
<img width="2894" height="442" alt="image" src="https://github.com/user-attachments/assets/73ab5e67-b9e8-4695-bbe5-b2450ce0b626" />

---

# 📌 Project Overview

Sports complexes often receive repetitive questions regarding:

* Membership plans
* Sports facilities
* Booking procedures
* Operating hours
* Safety policies
* Coaching staff
* Rules and regulations

Manually handling these inquiries can be time-consuming and inefficient.

Sportify AI automates this process through a Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant information from internal documents and generates accurate responses in real time.

---

# ✨ Features

## 💬 Natural Language Q&A

Users can ask questions naturally:

```text
What are the membership options?
```

```text
Can I book a badminton court online?
```

```text
What are the swimming pool timings?
```

The chatbot responds using information retrieved from the knowledge base.

---

## 🧠 Retrieval-Augmented Generation (RAG)

Rather than relying entirely on an LLM's internal knowledge:

1. User asks a question
2. Relevant documents are retrieved using vector search
3. Retrieved context is sent to the LLM
4. The LLM generates a grounded response

This approach improves:

✅ Accuracy

✅ Reliability

✅ Context-awareness

✅ Reduced hallucinations

---

## 🔍 Semantic Search

The system understands meaning, not just keywords.

For example:

```text
How much does membership cost?
```

and

```text
What are your membership plans?
```

can retrieve similar information despite different wording.

---

## 📚 Knowledge Base Integration

The chatbot retrieves information from multiple organizational documents:

* Sports Information
* Facility Overview
* Timings
* Membership Plans
* Booking Rules
* Safety Policies
* Coaches Information
* Member Guidelines
* FAQs

---

## 🗣️ Conversational Memory

The chatbot maintains chat history during the session, enabling more natural multi-turn conversations.

Example:

```text
User: What memberships do you offer?

Assistant: [Provides plans]

User: Which one is best for families?
```

The AI can use previous context to continue the conversation.

---

## ⚡ Robust API Handling

Includes retry logic for:

* Temporary network interruptions
* API connection failures
* Service instability

Ensuring a smoother user experience.

---

# 🏗️ System Architecture

```text
User Query
     │
     ▼
Sentence Transformer
(Query Embedding)
     │
     ▼
FAISS Vector Search
     │
Retrieve Top-K Chunks
     │
     ▼
Context Assembly
     │
     ▼
Groq LLM
(Llama 3.1 8B Instant)
     │
     ▼
Generated Response
```

---

# 🛠️ Tech Stack

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Core Programming Language           |
| Streamlit             | Web Application Framework           |
| Groq API              | LLM Inference                       |
| Llama 3.1 8B Instant  | Response Generation                 |
| Sentence Transformers | Text Embeddings                     |
| all-MiniLM-L6-v2      | Embedding Model                     |
| FAISS                 | Vector Database & Similarity Search |
| NumPy                 | Numerical Processing                |

---

# 📂 Project Structure

```bash
Sportify-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── overview.txt
│   ├── sports.txt
│   ├── timings.txt
│   ├── memberships.txt
│   ├── booking_rules.txt
│   ├── safety.txt
│   ├── coaches.txt
│   ├── members.txt
│   └── faqs.txt
```

---

# ⚙️ How It Works

## Step 1: Knowledge Base Loading

All text files are loaded from the data directory.

```python
load_documents()
```

---

## Step 2: Text Chunking

Documents are split into smaller chunks to improve retrieval quality.

```python
chunk_text()
```

Benefits:

* Better semantic search
* Faster retrieval
* More precise context selection

---

## Step 3: Embedding Generation

Each chunk is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

Sentence Transformer embeddings capture semantic meaning rather than exact wording.

---

## Step 4: Vector Indexing

Embeddings are stored inside a FAISS vector index.

```python
faiss.IndexFlatL2
```

This enables efficient nearest-neighbor similarity search.

---

## Step 5: Retrieval

When a user submits a query:

1. Query is embedded
2. Top 5 most relevant chunks are retrieved
3. Retrieved context is assembled

---

## Step 6: Response Generation

The retrieved context is passed to:

```text
Llama 3.1 8B Instant
```

via Groq.

The model generates an answer grounded in the retrieved information.

---

# 🔍 Example Queries

### Membership Questions

```text
What membership plans are available?
```

---

### Booking Questions

```text
How do I reserve a tennis court?
```

---

### Facility Questions

```text
What sports are available at the complex?
```

---

### Timings

```text
What time does the swimming pool open?
```

---

### Coaching Information

```text
Do you have professional football coaches?
```

---

# 🎯 Key AI Concepts Demonstrated

This project showcases several important Generative AI concepts:

### Retrieval-Augmented Generation (RAG)

Combining retrieval systems with LLMs.

---

### Semantic Search

Finding information based on meaning rather than exact keywords.

---

### Embeddings

Transforming text into high-dimensional vector representations.

---

### Vector Databases

Using FAISS for similarity-based retrieval.

---

### Prompt Engineering

Structuring context and instructions to improve response quality.

---

### Conversational AI

Maintaining chat history for contextual interactions.

---

# 📈 Potential Future Improvements

* Voice-enabled assistant
* PDF knowledge base ingestion
* Admin dashboard
* User authentication
* Conversation export
* Hybrid search (Keyword + Vector)
* Metadata filtering
* Multi-language support
* Advanced RAG with reranking
* Retrieval evaluation metrics

---

# 🎓 Learning Outcomes

Through this project, I gained hands-on experience with:

* Building end-to-end RAG systems
* Vector databases using FAISS
* Embedding models and semantic search
* Groq LLM integration
* Streamlit deployment
* Conversational AI systems
* Knowledge-grounded response generation
* AI application architecture

---

# 👨‍💻 Author

**Ayesha Tariq**

Aspiring AI Engineer | Generative AI Developer | Building practical AI systems that solve real-world problems.

---

# 📄 License

This project is licensed under the MIT License.
