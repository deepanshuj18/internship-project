'''import os
import pinecone
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index("rag-chatbot-index")

# Example query embedding (replace with actual query if needed)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
query_vector = [0.0] * 1536  # Dummy vector just to see if index responds

result = index.query(vector=query_vector, top_k=5, include_metadata=True)
print("Matches returned from Pinecone:")
for match in result.matches:
    print(f"ID: {match.id}, Source: {match.metadata.get('source')}, Text snippet: {match.metadata.get('text')[:100]}...")'''


'''import os
import pinecone
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys
load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index("rag-chatbot-index")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Your actual query
query_text = "Tell me about Wattmonk's company overview and solar services"

# Create embedding for the query
query_vector = genai.embed_content(
    model="models/embedding-001",
    content=query_text,
    task_type="retrieval_document"
)["embedding"]

# Query Pinecone
result = index.query(vector=query_vector, top_k=3, include_metadata=True)

# Display results
print("Top matches from Pinecone:")
for match in result.matches:
    print(f"ID: {match.id}")
    print(f"Source: {match.metadata.get('source')}")
    print(f"Text snippet: {match.metadata.get('text')[:300]}...\n")'''




import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your index
index_name = "rag-chatbot-index"
if index_name not in [i.name for i in pc.list_indexes().indexes]:
    raise ValueError(f"Index '{index_name}' does not exist")
index = pc.Index(index_name)

# Configure Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Example query
query_text = "Tell me about Wattmonk's company overview and solar services"

# Create embedding using Google API
query_vector = genai.embed_content(
    model="models/embedding-001",
    content=query_text,
    task_type="retrieval_document"
)["embedding"]

# Query Pinecone
result = index.query(vector=query_vector, top_k=3, include_metadata=True)

# Display results
print("Top matches from Pinecone:")
for match in result.matches:
    print(f"ID: {match.id}")
    print(f"Source: {match.metadata.get('source')}")
    print(f"Text snippet: {match.metadata.get('text')[:300]}...\n")


