import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# LangChain / Groq imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# Load local .env (only used when developing locally)
load_dotenv()

# -----------------------------------------------------------------------------
# Page config & basic theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Toji – Resume Skill Scout",
    page_icon="🧬",
    layout="wide"
)

# Optional: custom CSS for a more modern look
st.markdown(
    """
    <style>
        .main {
            background-color: #050816;
            color: #f5f5f5;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .toji-title {
            font-size: 2.3rem;
            font-weight: 700;
            padding: 0;
            margin-bottom: 0.5rem;
        }
        .toji-subtitle {
            font-size: 0.95rem;
            color: #d1d5db;
            margin-bottom: 1.5rem;
        }
        .stFileUploader label {
            font-weight: 600;
        }
        .skill-pill {
            display: inline-block;
            background: #111827;
            color: #e5e7eb;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            border: 1px solid #1f2933;
        }
        .toji-badge {
            background: linear-gradient(135deg, #22c55e, #06b6d4);
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            color: #020617;
            font-weight: 600;
            margin-left: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header section
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([0.15, 0.85])
with col_logo:
    st.markdown("### 🧬")
with col_title:
    st.markdown('<div class="toji-title">Toji – Resume Skill Scout</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="toji-subtitle">Upload multiple resumes as PDFs and instantly search for candidates by skills, tools, or tech stacks.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Secure API key loading
# -----------------------------------------------------------------------------
groq_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = os.getenv("GROQ_API_KEY")

with st.sidebar:
    st.markdown("### ⚙️ Toji Settings")
    st.caption("Configure your **Groq** key and indexing options here.")
    st.write("Groq key loaded:", bool(groq_key))
    if groq_key:
        try:
            masked = "****" + groq_key[-4:]
        except Exception:
            masked = "****"
        st.caption(f"Groq key (masked): `{masked}`")
    else:
        st.error(
            "GROQ_API_KEY not found.\n\nAdd it to **Streamlit Secrets** or a local `.env` file."
        )

if not groq_key:
    st.stop()

# -----------------------------------------------------------------------------
# Setup LLM (Groq)
# -----------------------------------------------------------------------------
try:
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
    )
except Exception as e:
    st.error("Failed to initialize Groq LLM. Check your API key and network.")
    st.exception(e)
    st.stop()

# -----------------------------------------------------------------------------
# Layout: Left – Upload & Index, Right – Search
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([0.45, 0.55])

# Shared objects
embeddings = FakeEmbeddings(size=384)  # replace with real embeddings for production[web:13][web:19]
PERSIST_DIR = "chroma_db"

# -----------------------------------------------------------------------------
# Left: Upload and index resumes
# -----------------------------------------------------------------------------
with left_col:
    st.subheader("📂 Upload resumes")
    st.caption("Drop candidate resumes here. Toji will index them for semantic skill search.")

    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    auto_persist = st.checkbox(
        "Persist index locally (Chroma DB)",
        value=True,
        help="Stores the vector index on disk so you can reuse it without re-uploading.",
    )

    if uploaded_files:
        with st.spinner("Reading and indexing resumes..."):
            documents = []
            for pdf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf.getbuffer())
                    temp_path = tmp_file.name

                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                documents.extend(docs)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)

            vectordb = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory=PERSIST_DIR if auto_persist else None,
            )

            if auto_persist:
                vectordb.persist()

        st.success(f"✅ Indexed {len(uploaded_files)} resume(s) and {len(chunks)} chunk(s).")
        st.caption("You can now search for skills, tools, or roles on the right.")

# -----------------------------------------------------------------------------
# Right: Search skills and show candidates
# -----------------------------------------------------------------------------
with right_col:
    st.subheader("🎯 Search by skills")
    st.caption("Example: `Python, Django, REST APIs`, `Data Engineer`, `Java + Spring`.")

    if os.path.exists(PERSIST_DIR):
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

        skill_query = st.text_input(
            "Search for skills, tools, or roles",
            placeholder="e.g., Python, Java, ML, React, Data Engineer",
        )

        top_k = st.slider("Number of candidates to retrieve", 1, 10, 5)

        if skill_query:
            with st.spinner("Finding the best matching candidates..."):
                docs = vectordb.similarity_search(skill_query, k=top_k)
                if not docs:
                    st.warning("No matches found. Try different skills or upload more resumes.")
                else:
                    st.markdown("#### 📌 Matching Candidates")
                    context = "\n\n".join([d.page_content for d in docs])

                    prompt = f"""
                    You are Toji, an AI assistant that helps recruiters quickly shortlist candidates.

                    From the following resume content, identify the candidates that best match
                    these skills or requirements: {skill_query}.

                    For each suitable candidate, provide:
                    - Candidate tag or inferred name (if visible in the resume)
                    - Top 5–8 key skills or technologies
                    - 1–2 line suitability summary

                    Resume Content:
                    {context}
                    """

                    try:
                        result = llm.invoke(prompt)
                        text = getattr(result, "content", str(result))

                        # Nicely display the answer
                        st.markdown("###### Toji’s recommendations")
                        st.info(text)
                    except Exception as e:
                        st.error(
                            "Error while calling Groq API. Check your GROQ_API_KEY, quota, and network."
                        )
                        st.exception(e)
    else:
        st.info(
            "No index found yet. Upload resumes on the left panel to create a searchable index."
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Toji – Resume Skill Scout • Powered by Groq, LangChain, and Chroma. Replace FakeEmbeddings with a real embedding model for production use."
)
