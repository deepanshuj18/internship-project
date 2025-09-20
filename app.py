import streamlit as st
import os
import json
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
#from pinecone import Pinecone, ServerlessSpec
import pinecone
import numpy as np
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - NEC vs Wattmonk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

class RAGChatbot:
    def __init__(self):
        self.setup_api_keys()
        self.setup_gemini()
        self.setup_pinecone()
        
    def setup_api_keys(self):
        """Load API keys from environment variables"""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not self.google_api_key:
            st.error("❌ GOOGLE_API_KEY not found in environment variables")
            st.stop()
        if not self.pinecone_api_key:
            st.error("❌ PINECONE_API_KEY not found in environment variables")
            st.stop()
    
    def setup_gemini(self):
        """Initialize Google Gemini"""
        try:
            genai.configure(api_key=self.google_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            # Initialize embedding model for better vector generation
            self.embedding_model = genai.embed_content
            st.success("✅ Gemini initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini: {str(e)}")
            st.stop()
    
    '''def setup_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            # You'll need to replace 'your-index-name' with your actual index name
            self.index_name = "rag-chatbot-index"  # Update this with your actual index name
            self.index = self.pc.Index(self.index_name)
            st.success("✅ Pinecone initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
            st.stop()'''
    '''def setup_pinecone(self):
        try:
          
          pinecone.init(api_key=self.pinecone_api_key, environment=os.getenv("PINECONE_ENV"))
          self.index_name = "rag-chatbot-index"  # replace with your actual index name
          self.index = pinecone.Index(self.index_name)
          st.success("✅ Pinecone initialized successfully")
        except Exception as e:
          st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
          st.stop()'''
    def setup_pinecone(self):

        """Initialize Pinecone vector database using the new SDK"""
        try:
           
           from pinecone import Pinecone

        # Create Pinecone client
           pc = Pinecone(api_key=self.pinecone_api_key)

        # Connect to your index
           self.index_name = "rag-chatbot-index"  # replace with your actual index name
           self.index = pc.Index(self.index_name)

           st.success("✅ Pinecone initialized successfully")
        except Exception as e:

            st.error(f"❌ Failed to initialize Pinecone: {str(e)}")
            st.stop()

    def detect_intent(self, query: str) -> str:
        """Detect if query is about NEC, Wattmonk, or general"""
        query_lower = query.lower()
        
        # NEC keywords
        nec_keywords = ['nec', 'national electrical code', 'electrical code', 'wiring', 'installation', 'safety']
        # Wattmonk keywords  
        wattmonk_keywords = ['wattmonk', 'energy monitoring', 'power monitoring', 'smart meter', 'consumption']
        
        nec_score = sum(1 for keyword in nec_keywords if keyword in query_lower)
        wattmonk_score = sum(1 for keyword in wattmonk_keywords if query_lower)
        
        if nec_score > wattmonk_score and nec_score > 0:
            return "NEC"
        elif wattmonk_score > 0:
            return "Wattmonk"
        else:
            return "General"
    
    def retrieve_documents(self, query: str, intent: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents from Pinecone"""
        try:
            # Create query embedding using Google's embedding model
            query_vector = self.create_embedding(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter={"intent": intent} if intent != "General" else None
            )
            
            documents = []
            for match in results.matches:
                documents.append({
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "Unknown"),
                    "similarity": float(match.score),
                    "intent": match.metadata.get("intent", "General")
                })
            
            return documents
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
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
            st.warning(f"Using fallback embedding due to error: {str(e)}")
            # Fallback to simple embedding if Google embedding fails
            dimension = 768  # Standard dimension for embedding-001
            return np.random.random(dimension).tolist()
    
    def generate_response(self, query: str, documents: List[Dict], conversation_history: List[Dict]) -> str:
        """Generate response using Gemini with retrieved context"""
        try:
            # Prepare context from retrieved documents
            if documents:
                context = "\n\n".join([
                    f"Source: {doc['source']}\nContent: {doc['text']}\nSimilarity: {doc['similarity']:.2%}"
                    for doc in documents
                ])
                context_available = True
            else:
                context = "No relevant documents found in the knowledge base."
                context_available = False
            
            # Prepare conversation history
            history_context = ""
            if conversation_history:
                history_context = "\n\nPrevious conversation:\n"
                for msg in conversation_history[-3:]:  # Last 3 messages
                    history_context += f"{msg['role']}: {msg['content']}\n"
            
            # Create prompt based on whether context is available
            if context_available:
                prompt = f"""
                You are a helpful assistant specializing in electrical engineering, NEC (National Electrical Code), and energy monitoring systems like Wattmonk.
                
                {history_context}
                
                Context from knowledge base:
                {context}
                
                User Question: {query}
                
                Please provide a comprehensive answer based on the context provided. Always cite your sources when possible.
                If the context doesn't fully answer the question, mention what additional information might be helpful.
                """
            else:
                prompt = f"""
                You are a helpful assistant specializing in electrical engineering, NEC (National Electrical Code), and energy monitoring systems like Wattmonk.
                
                {history_context}
                
                User Question: {query}
                
                I don't have specific documents in my knowledge base to answer this question, but I can provide general guidance based on my training.
                Please note that for specific code requirements or technical specifications, you should consult the official NEC documentation or contact a licensed electrician.
                """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_follow_up_questions(self, query: str, response: str, documents: List[Dict]) -> List[str]:
        """Generate follow-up questions based on the conversation"""
        try:
            prompt = f"""
            Based on this conversation:
            User Question: {query}
            Assistant Response: {response}
            
            Generate 2-3 relevant follow-up questions that the user might want to ask next.
            Make them specific and helpful.
            Return only the questions, one per line, without numbering.
            """
            
            follow_up_response = self.model.generate_content(prompt)
            questions = [q.strip() for q in follow_up_response.text.split('\n') if q.strip()]
            return questions[:3]  # Return max 3 questions
        except Exception as e:
            return ["Could you provide more details about this topic?", "What are the key considerations for this?"]
    
    def process_query(self, query: str) -> Dict:
        """Process a user query and return complete response"""
        # Detect intent
        intent = self.detect_intent(query)
        
        # Retrieve documents
        documents = self.retrieve_documents(query, intent)
        
        # Generate response
        response = self.generate_response(query, documents, st.session_state.conversation_history)
        
        # Generate follow-up questions
        follow_up_questions = self.generate_follow_up_questions(query, response, documents)
        
        return {
            "query": query,
            "intent": intent,
            "response": response,
            "documents": documents,
            "follow_up_questions": follow_up_questions
        }

def main():
    st.title("🤖 RAG Chatbot - NEC vs Wattmonk")
    st.markdown("Ask questions about electrical engineering, NEC codes, or energy monitoring systems!")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        st.markdown("### Intent Detection")
        st.info("The chatbot automatically detects if your question is about:\n- **NEC**: National Electrical Code\n- **Wattmonk**: Energy monitoring systems\n- **General**: Other electrical topics")
        
        st.markdown("### Confidence Scoring")
        st.info("Each retrieved document shows a similarity score indicating how relevant it is to your query.")
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.markdown(f"**Similarity:** {source['similarity']:.2%}")
                        st.markdown(f"**Content:** {source['text'][:200]}...")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about electrical engineering..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chatbot.process_query(prompt)
            
            # Display response
            st.markdown(result["response"])
            
            # Display intent
            intent_colors = {"NEC": "🔵", "Wattmonk": "🟢", "General": "⚪"}
            st.markdown(f"**Detected Intent:** {intent_colors.get(result['intent'], '⚪')} {result['intent']}")
            
            # Display sources
            if result["documents"]:
                with st.expander("📚 Sources & Confidence Scores"):
                    for i, doc in enumerate(result["documents"], 1):
                        st.markdown(f"**Source {i}:** {doc['source']}")
                        st.markdown(f"**Confidence:** {doc['similarity']:.2%}")
                        st.markdown(f"**Content:** {doc['text'][:300]}...")
                        st.markdown("---")
            else:
                st.warning("⚠️ No relevant documents found in the knowledge base.")
            
            # Display follow-up questions
            if result["follow_up_questions"]:
                st.markdown("**💡 Suggested Follow-up Questions:**")
                for i, question in enumerate(result["follow_up_questions"], 1):
                    if st.button(f"{i}. {question}", key=f"followup_{i}_{len(st.session_state.messages)}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["response"],
            "sources": result["documents"],
            "intent": result["intent"]
        })
        
        # Update conversation history
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "assistant", "content": result["response"]})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()
