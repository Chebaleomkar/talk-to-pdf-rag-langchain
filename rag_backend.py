import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from config import INDEX_NAME

load_dotenv()


class PDFRAGSystem:
    """RAG system for PDF question answering with deduplication"""
    
    def __init__(self):
        """Initialize the RAG system - runs once when you create the object"""
        print("🚀 Initializing PDF RAG System...")
        
        self.index_name = INDEX_NAME
        self.pc = self._initialize_pinecone()
        self.embeddings = self._initialize_embeddings()
        print("✅ System initialized!\n")
    
    def _initialize_pinecone(self):
        """Private method to setup Pinecone (called only once in __init__)"""
        api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)
        
        print("✅ Pinecone connected")
        
        # Check and create index if needed
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"📊 Creating index: '{self.index_name}'")
            pc.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("✅ Index created")
        else:
            print(f"✅ Index '{self.index_name}' exists")
        
        return pc
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash for duplicate detection"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(1024 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _initialize_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True}
        )
        return embeddings
    
    def load_chunk_pdf(self,pdf_path):
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split priority: paragraphs → lines → words → characters
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        file_hash = self.calculate_file_hash(pdf_path)
        filename = os.path.basename(pdf_path)

        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] =i
            chunk.metadata['source_file']=filename
            chunk.metadata["file_hash"] = file_hash 

        return chunks,file_hash
    
    def is_pdf_already_processed(self,file_hash):
        from langchain_pinecone import PineconeVectorStore
        
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        index = self.pc.Index(self.index_name)

        dummy_vector=[0.0]*384

        results = index.query(
            vector=dummy_vector,
            top_k=1,
            include_metadata=True,
            filter={"file_hash": {"$eq": file_hash}}
        )
        
        if results['matches']:
            print("✅ PDF already processed")
            return True
        return False

    def upload_chunks_to_vdb(self,chunks):
        from langchain_pinecone import PineconeVectorStore

        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,  # Uses the model we initialized
            index_name=self.index_name
        )
        print("✅ Chunks uploaded to Pinecone") 
        return vector_store
    def query(self, question):
        from langchain_pinecone import PineconeVectorStore
        from langchain_groq import ChatGroq
        # FIXED: For langchain 1.x, these are in langchain_core
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        print(f"\n❓ {question}")
        print("🔍 Searching knowledge base...")

        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Template for RAG
        template = """You are a helpful AI assistant that answers questions based ONLY on the provided context from PDF documents.

    Context from documents:
    {context}

    IMPORTANT RULES:
    1. Answer ONLY using information from the context above
    2. ALWAYS cite sources like this: [Page X]
    3. If context doesn't contain the answer, say "I don't have this information in the provided documents."
    4. Be concise but complete

    Question: {question}

    Answer with citations:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join([f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" for doc in docs])
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Get answer
        answer = rag_chain.invoke(question)
        
        # Get source documents separately
        source_docs = retriever.invoke(question)
        
        return {
            "answer": answer,
            "sources": source_docs
        }
   
if __name__ == "__main__":
    # Initialize system
    rag = PDFRAGSystem()
    
    # Upload PDF (first time only)
    pdf_path = "attention-is-all-you-need-Paper.pdf"
    if os.path.exists(pdf_path):
        chunks, file_hash = rag.load_chunk_pdf(pdf_path)
        
        if not rag.is_pdf_already_processed(file_hash):
            rag.upload_chunks_to_vdb(chunks)
            print("✅ PDF uploaded!\n")
        
        # Now query the system
        print("\n" + "="*50)
        print("🤖 Testing RAG Query")
        print("="*50)
        
        question = "What is the main topic of this document?"
        result = rag.query(question)
        
        print(f"💡 ANSWER:\n{result['answer']}\n")