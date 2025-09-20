"""
Deployment script for the RAG Chatbot.
This script helps set up the environment and deploy the application.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy env_template.txt to .env and fill in your API keys")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_keys = ["GOOGLE_API_KEY", "PINECONE_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if f"{key}=" not in content or f"{key}=your_" in content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing or incomplete API keys: {', '.join(missing_keys)}")
        print("Please update your .env file with valid API keys")
        return False
    
    print("✅ .env file configured correctly")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e.stderr}")
        return False

def setup_pinecone():
    """Set up Pinecone index with sample data"""
    print("🗄️ Setting up Pinecone index...")
    try:
        subprocess.run([sys.executable, "sample_data_ingestion.py"], 
                      check=True, capture_output=True, text=True)
        print("✅ Pinecone index set up successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set up Pinecone: {e.stderr}")
        return False

def run_application():
    """Run the Streamlit application"""
    print("🚀 Starting the RAG Chatbot...")
    print("The application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run application: {e}")

def main():
    """Main deployment function"""
    print("🤖 RAG Chatbot Deployment Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Set up Pinecone
    if not setup_pinecone():
        print("⚠️ Pinecone setup failed, but you can still run the app")
        print("Make sure to run 'python sample_data_ingestion.py' manually")
    
    # Run application
    run_application()

if __name__ == "__main__":
    main()
