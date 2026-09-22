import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate


class EnterpriseRAGBackend:
    def __init__(self, collection_name: str = "enterprise_knowledge_base", persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.vector_store = self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self) -> Chroma:
        """Loads or creates the ChromaDB vector database."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def scrape_url(self, url: str) -> str:
        """Scrapes text content cleanly from a URL."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()

        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        return "\n".join(lines)

    def ingest_url(self, url: str) -> int:
        """Scrapes, chunks, and indexes a web URL into ChromaDB."""
        content = self.scrape_url(url)
        doc = Document(page_content=content, metadata={"source": url, "type": "web_page"})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def ingest_raw_text(self, text_content: str, title: str) -> int:
        """Processes raw text/document uploads into ChromaDB."""
        doc = Document(page_content=text_content, metadata={"source": title, "type": "uploaded_doc"})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents([doc])
        
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def answer_query(self, query_str: str, k: int = 3) -> Dict[str, Any]:
        """Retrieves contexts, performs grounded RAG generation, and returns citations."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query_str)

        if not relevant_docs:
            return {
                "answer": "No relevant context was found in the knowledge base.",
                "sources": []
            }

        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in relevant_docs]))

        system_prompt = (
            "You are ChromaQuery Enterprise AI Assistant. Answer the user query based ONLY on the context provided below. "
            "If the answer cannot be determined from the context, state that clearly without making up facts.\n\n"
            "Retrieved Context:\n{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        formatted_prompt = prompt_template.format_messages(context=context, question=query_str)
        response = self.llm.invoke(formatted_prompt)

        return {
            "answer": response.content,
            "sources": sources
        }
