import os
import pickle
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.tools import tool
from src.models.llm_config import get_embeddings

class CourseRetriever:

    def __init__(self, docs: List[Document], persist_path: str = "./faiss_store"):
        self.persist_path = persist_path
        self.vectorstore = self._build_vector_store(docs)

    def _build_vector_store(self, docs: List[Document]) -> FAISS:
        os.makedirs(self.persist_path, exist_ok=True)

        embeddings = get_embeddings()
        faiss_index_path = os.path.join(self.persist_path, "index")
        doc_store_path = os.path.join(self.persist_path, "doc_store.pkl")

        if os.path.exists(faiss_index_path) and os.path.exists(doc_store_path):
            try:
                print("[INFO] Loading existing FAISS index...")
                return FAISS.load_local(
                    faiss_index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[WARN] Failed to load FAISS index: {e}")

        print("[INFO] Creating new FAISS index...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        split_docs = splitter.split_documents(docs)

        if split_docs:
            print("[DEBUG] Sample document chunk:", split_docs[0])

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(faiss_index_path)

        with open(doc_store_path, "wb") as f:
            pickle.dump(split_docs, f)

        print("[INFO] FAISS index created and saved")
        return vectorstore

    def search_courses(self, query: str, k: int = 4) -> List[Document]:
        """Search for relevant courses for recommendation."""
        results = self.vectorstore.similarity_search(query, k=k)

        if not results:
            print("[INFO] No relevant courses found for query:", query)
        else:
            print(f"[INFO] Found {len(results)} course(s) for query:", query)

        return results
