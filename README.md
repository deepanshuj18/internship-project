# RAG Chatbot - NEC vs Wattmonk

A complete Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Google Gemini, and Pinecone. The chatbot specializes in electrical engineering topics, particularly NEC (National Electrical Code) and Wattmonk energy monitoring systems.

## 🚀 Features

- **Multi-turn Chat**: Maintains conversation history for context-aware responses
- **Intent Detection**: Automatically detects if queries are about NEC, Wattmonk, or general electrical topics
- **Document Retrieval**: Retrieves relevant documents from Pinecone vector database
- **Confidence Scoring**: Shows similarity scores for each retrieved document
- **Query Refinement**: Suggests 2-3 follow-up questions after each response
- **Source Display**: Shows sources and confidence scores for transparency
- **Fallback Handling**: Graceful handling when no relevant documents are found
- **Deployment Ready**: Includes all necessary files for easy deployment

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (gemini-1.5-flash)
- **Vector Database**: Pinecone
- **Embeddings**: Google's embedding-001 model
- **Environment**: Python 3.8+

## 📋 Prerequisites

1. **Google API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Pinecone Account**: Sign up at [Pinecone](https://app.pinecone.io/) and get your API key
3. **Python 3.8+**: Make sure you have Python installed

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/deepanshuj18/internship-project
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy the template
cp env_template.txt .env
```

Edit `.env` with your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-chatbot-index
```

### 3. Populate Pinecone Index

Run the sample data ingestion script to populate your Pinecone index with sample documents:

```bash
python sample_data_ingestion.py
```

This will:
- Create a Pinecone index if it doesn't exist
- Add sample documents about NEC, Wattmonk, and general electrical topics
- Test the retrieval system

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
rag-chatbot/
├── app.py                          # Main Streamlit application
├── sample_data_ingestion.py        # Script to populate Pinecone with sample data
├── requirements.txt                 # Python dependencies
├── env_template.txt                # Environment variables template
├── README.md                       # This file
└── .env                           # Your API keys (create this)
```

## 🔧 Configuration

### Pinecone Index Setup

The application expects a Pinecone index with the following configuration:
- **Dimension**: 768 (for Google's embedding-001 model)
- **Metric**: cosine
- **Cloud**: AWS
- **Region**: us-east-1

### Customizing Intent Detection

You can modify the intent detection logic in the `detect_intent()` method in `app.py`:

```python
def detect_intent(self, query: str) -> str:
    query_lower = query.lower()
    
    # Add your custom keywords here
    nec_keywords = ['nec', 'national electrical code', ...]
    wattmonk_keywords = ['wattmonk', 'energy monitoring', ...]
    
    # Your detection logic
```

## 📊 Adding Your Own Documents

To add your own documents to the Pinecone index, modify the `sample_data_ingestion.py` script:

```python
def add_your_documents(self):
    your_docs = [
        {
            "text": "Your document content here...",
            "source": "Document Source",
            "intent": "NEC"  # or "Wattmonk" or "General"
        },
        # Add more documents...
    ]
    
    for doc in your_docs:
        self.add_document(doc["text"], doc["source"], doc["intent"])
```

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your environment variables in the Streamlit Cloud dashboard
4. Deploy!

### Other Platforms

The application can be deployed to any platform that supports Python applications:

- **Heroku**: Add a `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Docker**: Create a Dockerfile and deploy to any container platform
- **AWS/GCP/Azure**: Deploy as a web service

## 🔍 Usage

1. **Ask Questions**: Type your electrical engineering questions in the chat input
2. **View Sources**: Click on the "Sources" expander to see retrieved documents and confidence scores
3. **Follow-up Questions**: Use the suggested follow-up questions for deeper exploration
4. **Intent Detection**: The chatbot automatically detects if your question is about NEC, Wattmonk, or general topics

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your `.env` file is properly configured
2. **Pinecone Connection**: Verify your Pinecone API key and index name
3. **Embedding Errors**: Check if your Google API key has access to the embedding model
4. **No Documents Found**: Run the sample data ingestion script to populate your index

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export STREAMLIT_DEBUG=true
streamlit run app.py
```

## 📈 Performance Optimization

- **Caching**: The application uses Streamlit's built-in caching for API calls
- **Batch Processing**: Consider batching document uploads for large datasets
- **Index Optimization**: Monitor your Pinecone index performance and adjust as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure all API keys are correctly configured
4. Verify your Pinecone index is properly set up

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (PDF, DOCX, etc.)
- [ ] Advanced query preprocessing
- [ ] User authentication and conversation persistence
- [ ] Integration with more LLM providers
- [ ] Advanced analytics and usage tracking
# internship-project
