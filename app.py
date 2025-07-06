# app.py - Production Ready Version
import os
import logging
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format=%(asctime)s %(levelname)s %(message)s,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(rag_app)

# ─── Helper Functions ────────────────────────────────────────────────────────

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)
    Load PDF and split into chunks
    logger.info(fLoading PDF {pdf_path})
    if not os.path.exists(pdf_path)
        raise FileNotFoundError(fPDF file not found {pdf_path})
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    if not pages
        raise ValueError(fNo content found in PDF {pdf_path})
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(pages)
    logger.info(fSplit into {len(docs)} chunks)
    return docs

def build_faiss_index(docs, embed_model, index_path=faiss_index)
    Build and save FAISS index
    if os.path.isdir(index_path)
        import shutil
        shutil.rmtree(index_path)
        logger.info(Removed existing FAISS index)
    
    logger.info(Building FAISS index...)
    vs = FAISS.from_documents(docs, embed_model)
    vs.save_local(index_path)
    logger.info(FAISS index built and saved)
    return vs

def create_qa_chain(vs, chat_model, k=3)
    Create RetrievalQA chain
    logger.info(fCreating RetrievalQA chain with k={k})
    retriever = vs.as_retriever(search_kwargs={k k})
    
    chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type=stuff,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    logger.info(RetrievalQA chain ready)
    return chain

# ─── Configuration and Initialization ───────────────────────────────────────

def initialize_app()
    Initialize all components
    load_dotenv()
    
    # Load environment variables
    ENDPOINT = os.getenv(AZURE_INFERENCE_ENDPOINT)
    API_KEY = os.getenv(AZURE_INFERENCE_KEY)
    EMBED_MODEL = os.getenv(AZURE_EMBED_MODEL, Text-Embedding-Model)
    CHAT_MODEL = os.getenv(AZURE_CHAT_MODEL, Chat-Completion-Model)
    PDF_PATH = os.getenv(PDF_FILE_PATH, Build.pdf)
    TEMPERATURE = float(os.getenv(CHAT_TEMPERATURE, 0.3))
    API_VERSION = os.getenv(AZURE_API_VERSION, 2024-12-01-preview)
    
    # Validate required environment variables
    if not ENDPOINT
        raise ValueError(AZURE_INFERENCE_ENDPOINT not found in environment variables)
    if not API_KEY
        raise ValueError(AZURE_INFERENCE_KEY not found in environment variables)
    
    logger.info(=  60)
    logger.info(INITIALIZING RAG APPLICATION)
    logger.info(=  60)
    logger.info(fEndpoint {ENDPOINT})
    logger.info(fEmbedding Model {EMBED_MODEL})
    logger.info(fChat Model {CHAT_MODEL})
    logger.info(fPDF Path {PDF_PATH})
    logger.info(fTemperature {TEMPERATURE})
    logger.info(fAPI Version {API_VERSION})
    
    try
        # Initialize embeddings
        logger.info(Initializing embedding model...)
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            azure_deployment=EMBED_MODEL,
            api_version=API_VERSION,
            chunk_size=1000
        )
        
        # Test embeddings
        logger.info(Testing embedding model...)
        test_embed = embeddings.embed_query(test embedding)
        logger.info(f✓ Embedding test successful - dimension {len(test_embed)})
        
        # Initialize chat model
        logger.info(Initializing chat model...)
        chat_model = AzureChatOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=API_KEY,
            azure_deployment=CHAT_MODEL,
            api_version=API_VERSION,
            temperature=TEMPERATURE,
            max_tokens=2000
        )
        
        # Test chat model
        logger.info(Testing chat model...)
        test_response = chat_model.invoke([
            HumanMessage(content=Say 'Hello, I am working correctly!')
        ])
        logger.info(f✓ Chat test successful {test_response.content})
        
        # Load and process PDF
        logger.info(Processing PDF...)
        docs = load_and_split_pdf(PDF_PATH)
        
        # Build vector store
        logger.info(Building vector store...)
        vs = build_faiss_index(docs, embeddings)
        
        # Create QA chain
        logger.info(Creating QA chain...)
        qa_chain = create_qa_chain(vs, chat_model)
        
        logger.info(=  60)
        logger.info(✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY!)
        logger.info(=  60)
        
        return {
            'embeddings' embeddings,
            'chat_model' chat_model,
            'qa_chain' qa_chain,
            'vector_store' vs
        }
        
    except Exception as e
        logger.error(❌ INITIALIZATION FAILED!)
        logger.error(fError {str(e)})
        logger.error(Please check)
        logger.error(1. Your environment variables are set correctly)
        logger.error(2. Your Azure deployments are active)
        logger.error(3. Your API key is valid)
        logger.error(4. Your PDF file exists)
        raise

# Initialize components globally
try
    components = initialize_app()
    embeddings = components['embeddings']
    chat_model = components['chat_model']
    qa_chain = components['qa_chain']
    vector_store = components['vector_store']
except Exception as e
    logger.critical(fFailed to initialize application {e})
    # Create dummy components to prevent import errors
    embeddings = None
    chat_model = None
    qa_chain = None
    vector_store = None

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)

@app.route()
def index()
    Serve the main page
    try
        # Try to serve the updated portfolio HTML file
        return send_file(index.html)
    except FileNotFoundError
        # Fallback HTML if index.html is not found
        return 
        html
        headtitleRAG Portfolio - Pankaj Shindetitlehead
        body style=font-family Arial, sans-serif; text-align center; padding 50px;
            h1🚀 RAG Portfolio is Live!h1
            pThe server is running successfully.p
            pstrongPortfolio file (index.html) not found.strongp
            pPlease upload your portfolio HTML file to see the complete portfolio.p
            pYou can test the API at codeapichatcodep
            hr
            pemPowered by Azure OpenAI & LangChainemp
        body
        html
        

@app.route(health)
def health()
    Health check endpoint
    status = {
        status healthy if qa_chain is not None else unhealthy,
        components {
            embeddings embeddings is not None,
            chat_model chat_model is not None,
            qa_chain qa_chain is not None,
            vector_store vector_store is not None
        },
        message RAG service is ready if qa_chain is not None else RAG service failed to initialize
    }
    return jsonify(status), 200 if qa_chain is not None else 500

@app.route(apichat, methods=[POST])
def chat()
    Main chat endpoint
    if qa_chain is None
        return jsonify({
            error Service not initialized properly. Check server logs.,
            suggestion Please contact the administrator.
        }), 503
    
    try
        # Get request data
        data = request.get_json(force=True)
        if not data
            return jsonify({error No JSON data provided}), 400
        
        query = data.get(query, ).strip()
        if not query
            return jsonify({error No query provided}), 400
        
        logger.info(fQuery received '{query}')
        
        # Handle simple greetings
        simple_greetings = {hi, hello, hey, hi there, hello there}
        if query.lower() in simple_greetings
            logger.info(Simple greeting detected - using direct chat model)
            try
                response = chat_model.invoke([
                    SystemMessage(content=You are a helpful assistant for Pankaj Shinde's portfolio. Introduce yourself and mention you can answer questions about his experience, skills, and projects.),
                    HumanMessage(content=query)
                ])
                return jsonify({
                    answer response.content,
                    type greeting
                })
            except Exception as e
                logger.error(fError in greeting response {e})
                return jsonify({
                    answer Hello! I'm Pankaj Shinde's AI assistant. I can help you learn about his experience, skills, and projects. What would you like to know,
                    type greeting
                })
        
        # Use RAG for other queries
        logger.info(Processing query with RAG pipeline...)
        result = qa_chain.invoke({query query})
        
        answer = result.get(result, I couldn't generate an answer.)
        source_docs = result.get(source_documents, [])
        
        logger.info(fRAG returned {len(source_docs)} source documents)
        
        # If no sources found, try direct chat model
        if not source_docs
            logger.info(No relevant documents found - using direct chat model)
            try
                response = chat_model.invoke([
                    SystemMessage(content=You are a helpful assistant representing Pankaj Shinde. The user asked a question but no relevant documents were found in the knowledge base. Provide a helpful response based on general knowledge about cloud architecture and professional services.),
                    HumanMessage(content=fUser question {query})
                ])
                return jsonify({
                    answer response.content,
                    type fallback,
                    note No specific information found in portfolio documents.
                })
            except Exception as e
                logger.error(fError in fallback response {e})
                return jsonify({
                    answer I couldn't find specific information about that in the portfolio documents. Please try asking about Pankaj's experience, skills, or specific projects.,
                    type error
                })
        
        return jsonify({
            answer answer,
            type rag
        })
        
    except Exception as e
        logger.exception(Error in apichat)
        return jsonify({
            error An error occurred while processing your request.,
            details str(e) if app.debug else Check server logs for details.
        }), 500

@app.route(apitest, methods=[GET])
def test_components()
    Test endpoint to verify all components
    if not all([embeddings, chat_model, qa_chain, vector_store])
        return jsonify({error Components not initialized}), 500
    
    try
        # Test embedding
        test_embed = embeddings.embed_query(test)
        
        # Test chat
        test_chat = chat_model.invoke([HumanMessage(content=Say 'test successful')])
        
        # Test retrieval
        retriever = vector_store.as_retriever()
        test_docs = retriever.get_relevant_documents(test query)
        
        return jsonify({
            status success,
            tests {
                embedding_dimension len(test_embed),
                chat_response test_chat.content,
                retrieval_docs len(test_docs)
            }
        })
    except Exception as e
        return jsonify({
            status error,
            error str(e)
        }), 500

@app.errorhandler(404)
def not_found(error)
    return jsonify({error Endpoint not found}), 404

@app.errorhandler(500)
def internal_error(error)
    return jsonify({error Internal server error}), 500

if __name__ == __main__
    if qa_chain is None
        logger.critical(Cannot start server - initialization failed!)
        exit(1)
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get(PORT, 8000))
    
    logger.info(f🚀 Starting Flask server on http0.0.0.0{port})
    logger.info(Available endpoints)
    logger.info(  - GET    Main portfolio page)
    logger.info(  - GET  health  Health check)
    logger.info(  - POST apichat  Chat endpoint)
    logger.info(  - GET  apitest  Test components)
    
    # Use Gunicorn-compatible settings for production
    app.run(host=0.0.0.0, port=port, debug=False)