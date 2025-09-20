"""
Test script to verify the RAG Chatbot setup.
Run this script to check if everything is configured correctly.
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        from pinecone import Pinecone
        print("✅ Pinecone imported successfully")
    except ImportError as e:
        print(f"❌ Pinecone import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_env_variables():
    """Test if environment variables are set correctly"""
    print("\n🔑 Testing environment variables...")
    
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not google_key or google_key == "your_google_api_key_here":
        print("❌ GOOGLE_API_KEY not set or using placeholder value")
        return False
    else:
        print("✅ GOOGLE_API_KEY is set")
    
    if not pinecone_key or pinecone_key == "your_pinecone_api_key_here":
        print("❌ PINECONE_API_KEY not set or using placeholder value")
        return False
    else:
        print("✅ PINECONE_API_KEY is set")
    
    return True

def test_gemini_connection():
    """Test Google Gemini API connection"""
    print("\n🤖 Testing Google Gemini connection...")
    
    try:
        import google.generativeai as genai
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test.")
        
        if response.text:
            print("✅ Google Gemini connection successful")
            return True
        else:
            print("❌ Google Gemini returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ Google Gemini connection failed: {e}")
        return False

def test_pinecone_connection():
    """Test Pinecone API connection"""
    print("\n🗄️ Testing Pinecone connection...")
    
    try:
        from pinecone import Pinecone
        load_dotenv()
        
        api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)
        
        # List indexes to test connection
        indexes = pc.list_indexes()
        print(f"✅ Pinecone connection successful. Found {len(indexes.names())} indexes")
        
        # Check if our index exists
        index_name = "rag-chatbot-index"
        if index_name in indexes.names():
            print(f"✅ Index '{index_name}' found")
            return True
        else:
            print(f"⚠️ Index '{index_name}' not found. Run sample_data_ingestion.py to create it.")
            return False
            
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation"""
    print("\n🔢 Testing embedding generation...")
    
    try:
        import google.generativeai as genai
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        
        # Test embedding generation
        result = genai.embed_content(
            model="models/embedding-001",
            content="This is a test document for embedding generation.",
            task_type="retrieval_document"
        )
        
        if result and 'embedding' in result and len(result['embedding']) > 0:
            print(f"✅ Embedding generation successful. Dimension: {len(result['embedding'])}")
            return True
        else:
            print("❌ Embedding generation failed - no embedding returned")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG Chatbot Setup Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Variables", test_env_variables),
        ("Google Gemini Connection", test_gemini_connection),
        ("Pinecone Connection", test_pinecone_connection),
        ("Embedding Generation", test_embedding_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("Run 'python deploy.py' or 'streamlit run app.py' to start the application.")
    else:
        print("⚠️ Some tests failed. Please check the errors above and fix them.")
        print("Make sure to:")
        print("1. Install all dependencies: pip install -r requirements.txt")
        print("2. Set up your .env file with valid API keys")
        print("3. Run sample_data_ingestion.py to populate Pinecone")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
