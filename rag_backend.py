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
        docuements = loader.load()
        print(f"Loaded {len(docuements)} pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            seperators=["\n\n", "\n", " ", ""]  # Split priority: paragraphs → lines → words → characters
        )
        chunks = text_splitter.split_documents(docuements)
        print(f"{len(chunks)} Chunks")

        file_hash = self.calculate_file_hash(pdf_path)
        filename = os.path.basename(pdf_path)

        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] =i
            chunk.metadata['source_file']=filename
            chunk.metadata["file_hash"] = file_hash 

        return chunks,file_hash
    

if __name__ == "__main__":
    # Initialize system
    rag = PDFRAGSystem()
    
    # Test PDF processing
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        chunks, file_hash = rag.load_chunk_pdf(pdf_path)
        
        # Show first chunk details
        print(f"\n📋 Sample Chunk:")
        print(f"Page: {chunks[0].metadata['page']}")
        print(f"Chunk ID: {chunks[0].metadata['chunk_id']}")
        print(f"Text preview: {chunks[0].page_content[:200]}...")

