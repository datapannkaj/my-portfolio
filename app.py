# app.py - ChromaDB Version for Render
import os
import logging
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_app")

# ─── Helper Functions ────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Load PDF and split into chunks"""
    logger.info(f"Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    if not pages:
        raise ValueError(f"No content found in PDF: {pdf_path}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(pages)
    logger.info(f"Split into {len(docs)} chunks")
    return docs

def build_vector_store(docs, embed_model, persist_dir="chroma_db"):
    """Build and save Chroma vector store"""
    import shutil
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info("Removed existing vector store")
    
    logger.info("Building vector store...")
    vs = Chroma.from_documents(
        documents=docs, 
        embedding=embed_model, 
        persist_directory=persist_dir
    )
    vs.persist()
    logger.info("Vector store built and saved")
    return vs

def create_qa_chain(vs, chat_model, k=3):
    """Create RetrievalQA chain"""
    logger.info(f"Creating RetrievalQA chain with k={k}")
    retriever = vs.as_retriever(search_kwargs={"k": k})
    
    chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    logger.info("RetrievalQA chain ready")
    return chain

# ─── Configuration and Initialization ───────────────────────────────────────

def initialize_app():
    """Initialize all components"""
    load_dotenv()
    
    # Load environment variables
    ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")
    API_KEY = os.getenv("AZURE_INFERENCE_KEY")
    EMBED_MODEL = os.getenv("AZURE_EMBED_MODEL", "Text-Embedding-Model")
    CHAT_MODEL = os.getenv("AZURE_CHAT_MODEL", "Chat-Completion-Model")
    PDF_PATH = os.getenv("PDF_FILE_PATH", "Build.pdf")
    TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.3))
    API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    
    # Validate required environment variables
    if not ENDPOINT:
        raise ValueError("AZURE_INFERENCE_ENDPOINT not found in environment variables")
    if not API_KEY:
        raise ValueError("AZURE_INFERENCE_KEY not found in environment variables")
    
    logger.info("=" * 60)
    logger.info("INITIALIZING RAG APPLICATION")
    logger.info("=" * 60)
    logger.info(f"Endpoint: {ENDPOINT}")
    logger.info(f"Embedding Model: {EMBED_MODEL}")
    logger.info(f"Chat Model: {CHAT_MODEL}")
    logger.info(f"PDF Path: {PDF_PATH}")
    
    try:
        # Initialize embeddings
        logger.info("Initializing embedding model...")
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            azure_deployment=EMBED_MODEL,
            api_version=API_VERSION
        )
        
        # Test embeddings
        logger.info("Testing embedding model...")
        test_embed = embeddings.embed_query("test embedding")
        logger.info(f"✓ Embedding test successful - dimension: {len(test_embed)}")
        
        # Initialize chat model
        logger.info("Initializing chat model...")
        chat_model = AzureChatOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            azure_deployment=CHAT_MODEL,
            api_version=API_VERSION,
            temperature=TEMPERATURE,
            max_tokens=2000
        )
        
        # Test chat model
        logger.info("Testing chat model...")
        test_response = chat_model.invoke([
            HumanMessage(content="Say 'Hello, I am working correctly!'")
        ])
        logger.info(f"✓ Chat test successful: {test_response.content}")
        
        # Load and process PDF
        logger.info("Processing PDF...")
        docs = load_and_split_pdf(PDF_PATH)
        
        # Build vector store
        logger.info("Building vector store...")
        vs = build_vector_store(docs, embeddings)
        
        # Create QA chain
        logger.info("Creating QA chain...")
        qa_chain = create_qa_chain(vs, chat_model)
        
        logger.info("=" * 60)
        logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return {
            'embeddings': embeddings,
            'chat_model': chat_model,
            'qa_chain': qa_chain,
            'vector_store': vs
        }
        
    except Exception as e:
        logger.error("❌ INITIALIZATION FAILED!")
        logger.error(f"Error: {str(e)}")
        raise

# Initialize components globally
try:
    components = initialize_app()
    embeddings = components['embeddings']
    chat_model = components['chat_model']
    qa_chain = components['qa_chain']
    vector_store = components['vector_store']
except Exception as e:
    logger.critical(f"Failed to initialize application: {e}")
    embeddings = None
    chat_model = None
    qa_chain = None
    vector_store = None

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.route("/")
def index():
    """Serve the main page"""
    try:
        return send_file("index.html")
    except FileNotFoundError:
        return """
        <html>
        <head><title>RAG Portfolio - Pankaj Shinde</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
            <h1>🚀 RAG Portfolio is Live!</h1>
            <p>The server is running successfully.</p>
            <p>You can test the API at <code>/api/chat</code></p>
        </body>
        </html>
        """

@app.route("/health")
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy" if qa_chain is not None else "unhealthy",
        "components": {
            "embeddings": embeddings is not None,
            "chat_model": chat_model is not None,
            "qa_chain": qa_chain is not None,
            "vector_store": vector_store is not None
        },
        "message": "RAG service is ready" if qa_chain is not None else "RAG service failed to initialize"
    }
    return jsonify(status), 200 if qa_chain is not None else 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    if qa_chain is None:
        return jsonify({
            "error": "Service not initialized properly. Check server logs."
        }), 503
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Query received: '{query}'")
        
        # Handle simple greetings
        simple_greetings = {"hi", "hello", "hey", "hi there", "hello there"}
        if query.lower() in simple_greetings:
            logger.info("Simple greeting detected")
            try:
                response = chat_model.invoke([
                    SystemMessage(content="You are Pankaj Shinde's AI assistant. Introduce yourself and mention you can answer questions about his experience, skills, and projects."),
                    HumanMessage(content=query)
                ])
                return jsonify({
                    "answer": response.content,
                    "type": "greeting"
                })
            except Exception as e:
                logger.error(f"Error in greeting response: {e}")
                return jsonify({
                    "answer": "Hello! I'm Pankaj Shinde's AI assistant. I can help you learn about his experience, skills, and projects. What would you like to know?",
                    "type": "greeting"
                })
        
        # Use RAG for other queries
        logger.info("Processing query with RAG pipeline...")
        result = qa_chain.invoke({"query": query})
        
        answer = result.get("result", "I couldn't generate an answer.")
        source_docs = result.get("source_documents", [])
        
        logger.info(f"RAG returned {len(source_docs)} source documents")
        
        if not source_docs:
            logger.info("No relevant documents found - using direct chat model")
            try:
                response = chat_model.invoke([
                    SystemMessage(content="You are representing Pankaj Shinde. The user asked a question but no relevant documents were found. Provide a helpful response based on general knowledge about cloud architecture."),
                    HumanMessage(content=f"User question: {query}")
                ])
                return jsonify({
                    "answer": response.content,
                    "type": "fallback"
                })
            except Exception as e:
                logger.error(f"Error in fallback response: {e}")
                return jsonify({
                    "answer": "I couldn't find specific information about that. Please try asking about Pankaj's experience, skills, or projects.",
                    "type": "error"
                })
        
        return jsonify({
            "answer": answer,
            "type": "rag"
        })
        
    except Exception as e:
        logger.exception("Error in /api/chat")
        return jsonify({
            "error": "An error occurred while processing your request."
        }), 500

if __name__ == "__main__":
    if qa_chain is None:
        logger.critical("Cannot start server - initialization failed!")
        exit(1)
    
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Flask server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)