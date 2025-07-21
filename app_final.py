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

load_dotenv()

st.set_page_config(
    page_title="AI PDF Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyCIooxFHkdrP1gpG8O3rQLBnKkr2ZZ6Ls8"
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

@st.cache_data
def process_pdf(file_content):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            return docs, chunks
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    except:
        return None, None

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = get_embeddings()
    return Chroma.from_documents(_chunks, embeddings)

def get_response(query, vectorstore):
    if not vectorstore:
        return "Please upload a PDF file first."
    
    try:
        model = configure_gemini()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Answer based on this PDF content:

{context}

Question: {query}
Answer:"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# CSS
st.markdown("""
<style>
.stApp { background-color: #2b2b2b; color: #ffffff; }
.user-msg { 
    background: #4a90e2; color: white; padding: 10px 15px; 
    border-radius: 15px 15px 5px 15px; margin: 5px 0 5px 30%; 
    max-width: 65%; display: block; 
}
.bot-msg { 
    background: #3a3a3a; color: white; padding: 10px 15px; 
    border-radius: 15px 15px 15px 5px; margin: 5px 30% 5px 0; 
    max-width: 65%; border: 1px solid #505050; display: block; 
}
.stTextInput input { 
    background: #3a3a3a; color: white; border: 2px solid #505050; 
    border-radius: 20px; 
}
.stButton button { 
    background: #4a90e2; color: white; border: none; border-radius: 20px; 
}
.stFileUploader > div > div { 
    background: #3a3a3a; border: 2px dashed #4a90e2; color: white; 
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar
with st.sidebar:
    st.title("💬 Chat History")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages:
        st.subheader("Recent Messages")
        for msg in st.session_state.messages[-5:]:
            if msg["role"] == "user":
                preview = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                with st.expander(f"Q: {preview}"):
                    st.write(msg["content"])

# Main interface
st.title("🤖 AI PDF Chat Assistant")

# File upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            try:
                file_content = uploaded_file.read()
                docs, chunks = process_pdf(file_content)
                
                if docs and chunks:
                    st.session_state.vectorstore = create_vectorstore(chunks)
                    st.success(f"✅ Processed {len(docs)} pages, {len(chunks)} chunks")
                else:
                    st.error("❌ Failed to process PDF")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Chat interface
st.subheader("💬 Chat")

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">You: {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">Assistant: {msg["content"]}</div>', unsafe_allow_html=True)

# Chat form to prevent infinite loop
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about your PDF:", key="question")
    submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        if st.session_state.vectorstore:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get response
            with st.spinner("Thinking..."):
                response = get_response(user_input, st.session_state.vectorstore)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()
        else:
            st.warning("⚠️ Please upload and process a PDF first!")