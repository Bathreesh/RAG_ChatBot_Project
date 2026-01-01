# RAG ChatBot Project – Resume Skill Search

A **RAG-based** (Retrieval-Augmented Generation) resume chatbot that lets users upload multiple PDF resumes and ask natural-language questions about candidates’ skills, experience, and suitability. The system combines semantic search over embedded documents with Groq-hosted LLMs to return accurate, context-aware answers grounded in the uploaded resumes. [page:1]

---

## 🚀 Live Demo

Try the deployed application here:

👉 **Streamlit App:** https://ragchatbotproject-9xnfqun5iyhsmzynanzrij.streamlit.app/ [page:1]

---

## 💡 Overview

This project focuses on resume analysis and skill search using RAG:

- Users upload one or more PDF resumes, which are parsed into text chunks.  
- Embeddings are created from these chunks and stored in a vector database (ChromaDB).  
- When the user asks a question, the most relevant chunks are retrieved and passed, along with the query, to a Groq LLM (Llama 3) via LangChain to generate grounded responses. [page:1]

Use cases include:

- Quickly identifying candidates who match specific technologies or roles.  
- Summarizing experience across multiple resumes.  
- Asking targeted questions such as “Who has the strongest Java + Spring Boot experience?” or “Which candidate has leadership or team management experience?”. [page:1]

---

## 🧱 Tech Stack

- **Language:** Python  
- **Framework:** Streamlit (web UI and file upload)  
- **Orchestration:** LangChain for RAG pipelines and prompt handling  
- **LLM Provider:** Groq (Llama 3 model family)  
- **Vector Store:** ChromaDB for embedding storage and similarity search  
- **Embeddings:** HuggingFace text embedding models [page:1]

---

## ✨ Features

- Upload multiple PDF resumes in a single session.  
- Automatic text extraction, splitting, embedding, and indexing for semantic search.  
- Query resumes using natural language (skills, tools, roles, experience, etc.).  
- RAG pipeline returns answers explicitly grounded in the most relevant resume chunks.  
- Streamlit-based interface that shows both the answer and the underlying candidate context (depending on how `app.py` is configured). [page:1]

---

## 📦 Project Structure
[page:1]

- `app.py`: Implements the RAG pipeline (file upload, embedding, ChromaDB indexing, and Groq-powered answer generation) and the Streamlit UI.  
- `requirements.txt`: Lists Python dependencies such as `streamlit`, `langchain`, `chromadb`, `huggingface-hub` / embeddings, and Groq client libraries. [page:1]

---

## ⚙️ Installation

Follow these steps to run the project locally.

1. **Clone the repository**


2. **(Optional) Create and activate a virtual environment**


3. **Install dependencies**


4. **Configure environment variables**

Obtain your Groq API key and any other required keys (for example, HuggingFace token if needed) and set them as environment variables or in a `.env`/`secrets.toml` file.


5. **Run the Streamlit app**


Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser to access the chatbot. [page:1]

---

## 💬 How to Use

1. Open the app (locally or via the live demo link).  
2. Upload one or more PDF resumes using the file uploader.  
3. Wait for the app to index the resumes (embedding + ChromaDB storage).  
4. Ask questions such as:
- “Find candidates with strong Java, Spring Boot, and REST APIs.”  
- “Who has at least 3 years of Python + Django experience?”  
- “Which resume mentions leadership or team management?”  
5. Review the generated answer and, if the UI exposes it, the underlying resume snippets used to support the answer. [page:1]

---

## 🔧 Configuration & Customization

You can adapt the chatbot to your own recruitment or document-search workflows:

- **Model selection:** Change the Groq LLM (e.g., different Llama 3 variant) and temperature/top‑k parameters in `app.py`.  
- **Chunking strategy:** Adjust chunk size and overlap for better context retrieval based on typical resume length.  
- **Prompt template:** Customize the system and user prompts to control the tone, format, or strictness of grounding in the source documents.  
- **Embedding model:** Swap the HuggingFace embedding model for a larger or smaller variant depending on performance and accuracy needs. [page:1]

---

## 🌐 Deployment

This project is suitable for deployment on Streamlit Community Cloud or other hosting platforms:

1. Push your code to GitHub (repository: `Bathreesh/RAG_ChatBot_Project`).  
2. Create a new Streamlit app from this repository in Streamlit Community Cloud.  
3. Set environment variables such as `GROQ_API_KEY` (and any others) in the platform’s secrets/settings.  
4. Deploy and share the generated URL (for example, the live demo link above). [page:1]

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. To contribute:

1. Fork the repository.  
2. Create a feature branch for your changes.  
3. Commit with clear, descriptive messages.  
4. Open a pull request explaining the motivation and implementation details.

Improvements could include better ranking/reranking of candidates, UI enhancements, or integrations with ATS/HR tools. [page:1]

---

## 📄 License

Add an appropriate open-source license (for example, MIT License) in a `LICENSE` file and reference it here. Update this section once a license is chosen.

> Adjust folder/file names if you change them in the repository.

