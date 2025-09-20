"""
Sample data ingestion script for populating Pinecone index with sample documents.
This script demonstrates how to add documents to your Pinecone index for the RAG chatbot.
"""

import os
import json
from typing import List, Dict
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataIngestion:
    def __init__(self):
        self.setup_api_keys()
        self.setup_gemini()
        self.setup_pinecone()
    
    def setup_api_keys(self):
        """Load API keys from environment variables"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    def setup_gemini(self):
        """Initialize Google Gemini"""
        genai.configure(api_key=self.google_api_key)
        self.embedding_model = genai.embed_content
    
    def setup_pinecone(self):
        """Initialize Pinecone and create index if it doesn't exist"""
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "rag-chatbot-index"
        
        # Check if index exists, create if not
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Dimension for Google's embedding-001
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Index created successfully!")
        else:
            print(f"Index {self.index_name} already exists")
        
        self.index = self.pc.Index(self.index_name)
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using Google's embedding model"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return None
    
    def add_document(self, text: str, source: str, intent: str, doc_id: str = None):
        """Add a single document to the Pinecone index"""
        if not doc_id:
            doc_id = f"{intent}_{len(text)}_{hash(text) % 10000}"
        
        embedding = self.create_embedding(text)
        if embedding is None:
            print(f"Failed to create embedding for document: {source}")
            return False
        
        try:
            self.index.upsert(vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "source": source,
                    "intent": intent
                }
            }])
            print(f"Successfully added document: {source}")
            return True
        except Exception as e:
            print(f"Error adding document {source}: {str(e)}")
            return False
    
    def add_sample_documents(self):
        """Add sample documents for testing"""
        sample_docs = [
            {
                "text": "The National Electrical Code (NEC) requires that all electrical installations be installed in accordance with the code to ensure safety. Article 210 covers branch circuits and requires GFCI protection for outlets in bathrooms, kitchens, and outdoor areas.",
                "source": "NEC Article 210",
                "intent": "NEC"
            },
            {
                "text": "Wattmonk is an energy monitoring system that provides real-time power consumption data. It helps users track their electricity usage, identify high-consumption appliances, and optimize energy efficiency through detailed analytics and reporting.",
                "source": "Wattmonk Documentation",
                "intent": "Wattmonk"
            },
            {
                "text": "Electrical safety is paramount in any installation. Always turn off power at the breaker before working on electrical circuits. Use proper personal protective equipment (PPE) including insulated tools and safety glasses.",
                "source": "Electrical Safety Guidelines",
                "intent": "General"
            },
            {
                "text": "NEC Article 250 covers grounding and bonding requirements. All electrical systems must be properly grounded to prevent electrical shock and ensure safe operation. Ground fault circuit interrupters (GFCIs) must be installed in wet locations.",
                "source": "NEC Article 250",
                "intent": "NEC"
            },
            {
                "text": "Wattmonk sensors can be installed on individual circuits to monitor specific appliances or areas. The system provides historical data, peak demand analysis, and cost tracking to help users make informed decisions about their energy usage.",
                "source": "Wattmonk Installation Guide",
                "intent": "Wattmonk"
            },
            {
                "text": "Wire sizing is critical for electrical safety. The NEC provides ampacity tables to determine the correct wire size based on current load, ambient temperature, and installation conditions. Undersized wires can cause fires.",
                "source": "Electrical Installation Standards",
                "intent": "General"
            }
        ]
        
        print("Adding sample documents to Pinecone...")
        success_count = 0
        for i, doc in enumerate(sample_docs):
            if self.add_document(doc["text"], doc["source"], doc["intent"], f"sample_{i}"):
                success_count += 1
        
        print(f"Successfully added {success_count}/{len(sample_docs)} documents")
    
    def query_index(self, query: str, top_k: int = 3):
        """Query the index to test retrieval"""
        print(f"Querying: {query}")
        embedding = self.create_embedding(query)
        if embedding is None:
            print("Failed to create query embedding")
            return
        
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            print(f"Found {len(results.matches)} results:")
            for i, match in enumerate(results.matches, 1):
                print(f"{i}. Source: {match.metadata['source']}")
                print(f"   Similarity: {match.score:.2%}")
                print(f"   Intent: {match.metadata['intent']}")
                print(f"   Text: {match.metadata['text'][:100]}...")
                print()
        except Exception as e:
            print(f"Error querying index: {str(e)}")

def main():
    try:
        ingestion = DataIngestion()
        
        # Add sample documents
        ingestion.add_sample_documents()
        
        # Test queries
        print("\n" + "="*50)
        print("Testing queries:")
        print("="*50)
        
        test_queries = [
            "What are the NEC requirements for GFCI protection?",
            "How does Wattmonk help with energy monitoring?",
            "What safety precautions should I take when working with electricity?"
        ]
        
        for query in test_queries:
            ingestion.query_index(query)
            print("-" * 30)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have set up your .env file with the required API keys.")

if __name__ == "__main__":
    main()
