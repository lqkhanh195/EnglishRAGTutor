from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import os
from qdrant_client import QdrantClient

class KnowledgeBase:
    def __init__(self, embedding, docs_folder):
        self.docs_folder = docs_folder
        self.embedding = embedding
        self.qdrant_path = "D:\\EngRAG\\LiteratureIsEasy\\tmp\\local_qdrant"
        self.collection_name = "TOEIC"
        self.qdrant = None
    
    def load_content(self, docs_path):
        loader = PyPDFLoader(docs_path)
        docs = loader.load()

        return docs
    
    def split_text(self, doc):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_documents(doc)
    
    def modify_vector_store(self):
        if os.path.exists(self.qdrant_path):  # Collection exists            
            print("Loading existing Qdrant vector store...")
            client = QdrantClient(path=self.qdrant_path)
            self.qdrant = Qdrant(
                client=client,
                embeddings=self.embedding,
                # folder_path=self.qdrant_path,
                collection_name=self.collection_name
            )

        else:
            for i, file in enumerate(os.listdir(self.docs_folder)):
                doc_path = os.path.join(self.docs_folder, file)
                doc = self.load_content(doc_path)
                doc_splitted = self.split_text(doc)

                # Add new documents to Qdrant
                if i == 0:
                    self.qdrant = Qdrant.from_documents(doc_splitted, self.embedding, path=self.qdrant_path, collection_name=self.collection_name)
                else:
                    self.qdrant.add_documents(doc_splitted)

        return self.qdrant 