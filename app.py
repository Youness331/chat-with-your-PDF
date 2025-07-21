import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI PDF Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Gemini API
@st.cache_resource
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyCIooxFHkdrP1gpG8O3rQLBnKkr2ZZ6Ls8"
    if not api_key:
        st.error("⚠️ API key not found")
        st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Optimized PDF processing
@st.cache_data(show_spinner=False)
def process_pdf(file_content):
    """Process PDF file and return document chunks"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            if not docs:
                raise Exception("No content found in PDF")
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            
            if not chunks:
                raise Exception("Failed to create chunks from PDF")
            
            return docs, chunks
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        return None, None

# Optimized embeddings
@st.cache_resource
def get_embedding_model():
    """Initialize and cache embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def create_vector_store(_chunks):
    """Create and cache vector store"""
    embedder = get_embedding_model()
    vectorstore = Chroma.from_documents(_chunks, embedder)
    return vectorstore

def get_ai_response(query: str, vectorstore, model) -> str:
    """Generate AI response based on query and context"""
    if not vectorstore:
        return "Please upload a PDF file first."
    
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create optimized prompt
        prompt = f"""You are an AI assistant that answers questions based on PDF content.

Context from PDF:
{context}

User Question: {query}

Instructions:
- Answer based only on the provided context
- Be accurate and concise
- If information isn't available, say "I cannot find this information in the document"
- Provide specific details when available

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Advanced CSS styling
def load_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e1e;
        border-right: 1px solid #404040;
    }
    
    /* Message styling */
    .user-message {
        background-color: #4a90e2;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0 8px 25%;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(74, 144, 226, 0.3);
        display: block;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background-color: #3a3a3a;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 25% 8px 0;
        max-width: 70%;
        border: 1px solid #505050;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        display: block;
        word-wrap: break-word;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #505050;
        padding: 12px 20px;
        font-size: 16px;
        background: #3a3a3a;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 0.2rem rgba(74, 144, 226, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #357abd;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(74, 144, 226, 0.4);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: #3a3a3a;
        border-radius: 12px;
        border: 2px dashed #4a90e2;
        padding: 20px;
        text-align: center;
        color: #ffffff;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #2d5a2d;
        border: 1px solid #4caf50;
        color: #a5d6a7;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #5d2d2d;
        border: 1px solid #f44336;
        color: #ef9a9a;
        border-radius: 8px;
    }
    
    .stWarning {
        background-color: #5d4f2d;
        border: 1px solid #ff9800;
        color: #ffcc80;
        border-radius: 8px;
    }
    
    .stInfo {
        background-color: #2d4f5d;
        border: 1px solid #2196f3;
        color: #90caf9;
        border-radius: 8px;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Chat container */
    .chat-container {
        background: #333333;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #505050;
        min-height: 400px;
    }
    
    /* Sidebar improvements */
    .sidebar .element-container {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #3a3a3a;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Override Streamlit defaults for dark theme */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #4a90e2;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #3a3a3a;
        color: #ffffff;
    }
    
    /* File uploader text */
    .stFileUploader label {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with chat history"""
    with st.sidebar:
        st.markdown("# 💬 Chat History")
        
        # Document info
        if 'doc_info' in st.session_state:
            st.markdown("### 📄 Document Info")
            info = st.session_state.doc_info
            st.markdown(f"**Pages:** {info['pages']}")
            st.markdown(f"**Chunks:** {info['chunks']}")
            st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Chat history
        if st.session_state.messages:
            st.markdown("### Recent Conversations")
            for i, message in enumerate(reversed(st.session_state.messages[-10:])):
                if message["role"] == "user":
                    preview = message["content"][:50] + "..." if len(message["content"]) > 50 else message["content"]
                    with st.expander(f"💭 {preview}", expanded=False):
                        st.markdown(f"**Q:** {message['content']}")
                        if i > 0 and st.session_state.messages[-(i)]["role"] == "assistant":
                            st.markdown(f"**A:** {st.session_state.messages[-(i)]['content']}")
        else:
            st.info("🤖 No conversations yet. Upload a PDF and start chatting!")
        
        # App info
        st.markdown("---")
        st.markdown("### 🔧 About")
        st.markdown("**AI PDF Chat Assistant**")
        st.markdown("Powered by Google Gemini 1.5 Flash")
        st.markdown("Built with Streamlit & LangChain")

def render_main_interface():
    """Render main chat interface"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI PDF Chat Assistant</h1>
        <p>Upload your PDF and start an intelligent conversation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📁 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to start chatting with it",
        label_visibility="collapsed"
    )
    
    # Process uploaded file
    if uploaded_file and uploaded_file != st.session_state.get('current_file'):
        with st.spinner("🔄 Processing your PDF... Please wait"):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                
                if not file_content:
                    st.error("❌ File appears to be empty")
                    return
                
                docs, chunks = process_pdf(file_content)
                
                if docs is None or chunks is None:
                    st.error("❌ Failed to process PDF file. Please try a different PDF.")
                    return
                
                if len(chunks) == 0:
                    st.error("❌ No text content found in PDF")
                    return
                
                st.session_state.vectorstore = create_vector_store(chunks)
                st.session_state.current_file = uploaded_file
                st.session_state.doc_info = {
                    'pages': len(docs),
                    'chunks': len(chunks),
                    'filename': uploaded_file.name
                }
                
                st.success(f"✅ Successfully processed **{uploaded_file.name}** ({len(docs)} pages, {len(chunks)} chunks)")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.error("Please try uploading a different PDF file or check if the file is corrupted.")
                return
    
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            placeholder="Ask me anything about your PDF...",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 🚀", use_container_width=True, type="primary")
    
    # Handle user input - Fixed to prevent infinite loop
    if send_button and user_input.strip():
        if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload a PDF file first!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            with st.spinner("🤔 Thinking..."):
                model = configure_gemini()
                response = get_ai_response(user_input, st.session_state.vectorstore, model)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear the input and rerun
            st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Load custom CSS
    load_css()
    
    # Render UI components
    render_sidebar()
    render_main_interface()

if __name__ == "__main__":
    main()
