import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

import tempfile
import os
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Resume Skill Search", page_icon="📑")
st.title("📑 Resume Skill Search Bot (Groq)")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

embeddings = FakeEmbeddings(size=384)

PERSIST_DIR = "chroma_db"

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama-3.1-8b-instant",temperature=0.3)

if uploaded_files:
    documents = []

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.getbuffer())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()

    st.success("✅ Resumes indexed successfully!")


if os.path.exists(PERSIST_DIR):
    vectordb = Chroma(persist_directory=PERSIST_DIR,embedding_function=embeddings)
    query = st.text_input("Search skills (eg: Python, Java, ML)")
    if query:
        docs = vectordb.similarity_search(query, k=5)
        st.subheader("📌 Matching Candidates")

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        From the following resume content, identify candidates
        matching these skills: {query}

        Resume Content:
        {context}
        """

        result = llm.invoke(prompt)
        st.success(result.content)
