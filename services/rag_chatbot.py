import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Get absolute paths relative to backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(os.path.dirname(BASE_DIR), "documents")
VECTORDB_PATH = os.path.join(os.path.dirname(BASE_DIR), "vectordb", "faiss_index")

class RAGChatbot:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.vectorstore = None
        self.llm = None
        self.chat_history = []
        
    def load_documents(self) -> List:
        """Load all PDF documents from documents folder with page metadata"""
        loader = DirectoryLoader(
            DOCUMENTS_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        for doc in documents:
            if 'page' in doc.metadata:
                doc.metadata['page_number'] = doc.metadata['page'] + 1
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        
        return documents
    
    def create_vectorstore(self):
        """Create or load vector database with page indexing"""
        if os.path.exists(VECTORDB_PATH):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vectorstore = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
            return
        
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        for chunk in chunks:
            if 'page_number' not in chunk.metadata and 'page' in chunk.metadata:
                chunk.metadata['page_number'] = chunk.metadata['page'] + 1
            if 'source' in chunk.metadata and 'filename' not in chunk.metadata:
                chunk.metadata['filename'] = os.path.basename(chunk.metadata['source'])
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.vectorstore.save_local(VECTORDB_PATH)
    
    def initialize_llm(self):
        """Initialize Groq LLM"""
        if not self.llm:
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="openai/gpt-oss-20b",
                temperature=0.2,
                max_tokens=2048
            )
    
    def chat(self, query: str) -> Dict:
        """Process user query and return response with page references"""
        if not self.vectorstore:
            self.create_vectorstore()
        
        if not self.llm:
            self.initialize_llm()
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(query, k=5)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Build chat history string
        history_str = ""
        for msg in self.chat_history[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                history_str += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"
        
        # Medical prompt
        system_prompt = """You are a specialized medical AI assistant with expertise in cardiology, ECG interpretation, and clinical practice.

Provide concise, direct answers to the user's question using the provided medical literature context.

Guidelines:
- Answer only what is asked - be brief and focused
- Use evidence from the provided context
- Include specific clinical recommendations when relevant
- Use clear medical terminology
- Reference page numbers when citing sources
- If the context doesn't contain the answer, say so clearly

Keep responses concise and actionable."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"""Previous conversation:
{history_str}

Context from medical literature:
{context}

Current question: {query}

Provide a comprehensive medical response:""")
        ])
        
        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({"query": query})
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response.content))
        
        # Extract source metadata
        sources = []
        for doc in docs:
            source_info = {
                "filename": doc.metadata.get("filename", doc.metadata.get("source", "Unknown").split('/')[-1]),
                "page": doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
            }
            sources.append(source_info)
        
        return {
            "answer": response.content,
            "sources": sources
        }
    
    def reset_memory(self):
        """Clear conversation history"""
        self.chat_history = []
