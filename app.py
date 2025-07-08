import os
import tempfile
import streamlit as st
from typing import List

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough

# Set page configuration
st.set_page_config(
    page_title="Chat dengan Dokumen PDF Anda 📄",
    page_icon="📄",
    layout="wide"
)

# Initialize session states at the very beginning - FIXED
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Halo! Silakan unggah dokumen PDF Anda, dan saya akan membantu menjawab pertanyaan tentang dokumen tersebut.")
        ]
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "chain" not in st.session_state:
        st.session_state.chain = None

# Call initialization function
initialize_session_state()

# Display the title
st.title("Chat dengan Dokumen PDF Anda 📄")

# Sidebar for API key and chat history
with st.sidebar:
    st.header("Konfigurasi")
    google_api_key = "AIzaSyAf12tw3KOkJH-DxPAFickgHmI5pU8I9sI"
    
    # Add separator
    st.markdown("---")
    
    # Chat History Section
    st.header("Riwayat Chat 💬")
    
    # Add clear chat button
    if st.button("Bersihkan Riwayat Chat 🗑️"):
        # Reset all session state variables
        st.session_state.messages = [
            AIMessage(content="Halo! Silakan unggah dokumen PDF Anda, dan saya akan membantu menjawab pertanyaan tentang dokumen tersebut.")
        ]
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.session_state.vector_store = None
        st.session_state.chain = None
        st.rerun()
    
    # Display chat history with timestamps
    if len(st.session_state.messages) > 1:  # If there are messages beyond the welcome message
        st.markdown("#### Pesan Terakhir")
        for idx, msg in enumerate(st.session_state.messages[-5:]):  # Show last 5 messages
            if isinstance(msg, HumanMessage):
                with st.container():
                    st.markdown("**🧑 Anda:**")
                    st.info(msg.content)
            elif isinstance(msg, AIMessage):
                with st.container():
                    st.markdown("**🤖 AI:**")
                    st.success(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
            
            # Add a small divider between messages
            if idx < len(st.session_state.messages[-5:]) - 1:
                st.markdown("---")
    else:
        st.info("Belum ada riwayat chat. Mulai dengan mengajukan pertanyaan!")
    
    # Add information about chat history
    with st.expander("ℹ️ Tentang Riwayat Chat"):
        st.markdown("""
        - Menampilkan 5 pesan terakhir
        - 🧑 : Pesan Anda
        - 🤖 : Jawaban AI
        - Jawaban panjang akan disingkat
        """)

@st.cache_resource
def process_pdf(_file_content, _google_api_key: str):
    """
    Process the PDF file and create a vector store for efficient retrieval.
    This function is cached to prevent reprocessing the same file multiple times.
    """
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(_file_content)
        tmp_path = tmp_file.name

    try:
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        documents = text_splitter.split_documents(pages)

        # Create embeddings using Google's model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=_google_api_key
        )

        # Create and return the vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store

    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def create_chain(vector_store, google_api_key: str):
    """
    Create a retrieval chain that combines document retrieval with question answering.
    """
    # Initialize the language model
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.7
    )

    # Create a prompt template that includes chat history
    prompt = ChatPromptTemplate.from_template("""
    Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
    
    Riwayat Chat:
    {chat_history}
    
    Gunakan konteks berikut dan riwayat chat di atas untuk menjawab pertanyaan:
    {context}
    
    Pertanyaan: {input}
    
    Berikan jawaban yang jelas dan informatif. Jika Anda tidak dapat menjawab berdasarkan konteks yang diberikan,
    katakan bahwa Anda tidak dapat menemukan informasi yang relevan dalam dokumen.
    Gunakan riwayat chat untuk memberikan jawaban yang lebih kontekstual dan relevan.
    """)

    # Create a chain for processing documents
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 3}),
        document_chain
    )

    return retrieval_chain

def get_chat_history():
    """Get chat history from memory in the correct format"""
    try:
        if hasattr(st.session_state, 'memory') and st.session_state.memory:
            memory_vars = st.session_state.memory.load_memory_variables({})
            return memory_vars.get("chat_history", [])
        return []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []

# File uploader in the main area
uploaded_file = st.file_uploader(
    "Unggah dokumen PDF Anda:",
    type="pdf",
    help="Pilih file PDF yang ingin Anda analisis"
)

# Process the uploaded file if both file and API key are provided
if uploaded_file and google_api_key:
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Process the PDF and create the vector store only if not already processed
        if st.session_state.vector_store is None:
            with st.spinner("Memproses dokumen PDF..."):
                st.session_state.vector_store = process_pdf(file_content, google_api_key)
                st.session_state.chain = create_chain(st.session_state.vector_store, google_api_key)
                st.success("Dokumen berhasil diproses! Silakan ajukan pertanyaan Anda.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses dokumen: {str(e)}")
        st.stop()

    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

    # Chat input
    if question := st.chat_input("Ajukan pertanyaan tentang dokumen Anda"):
        # Ensure we have a valid chain
        if st.session_state.chain is None:
            st.error("Chain belum diinisialisasi. Silakan upload dokumen terlebih dahulu.")
        else:
            # Add user's question to chat history
            st.session_state.messages.append(HumanMessage(content=question))

            # Display user's question
            with st.chat_message("user"):
                st.write(question)

            # Generate and display response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                try:
                    # Get chat history for context
                    chat_history = get_chat_history()
                    
                    # Get response from the chain
                    response = st.session_state.chain.invoke({
                        "input": question,
                        "chat_history": chat_history
                    })
                    answer = response["answer"]
                    
                    # Update the placeholder with the response
                    response_placeholder.write(answer)
                    
                    # Add AI's response to chat history
                    st.session_state.messages.append(AIMessage(content=answer))
                    
                    # Save to conversation memory
                    if hasattr(st.session_state, 'memory') and st.session_state.memory:
                        st.session_state.memory.save_context(
                            {"input": question},
                            {"answer": answer}
                        )
                except Exception as e:
                    error_message = f"Maaf, terjadi kesalahan: {str(e)}"
                    response_placeholder.error(error_message)
                    st.session_state.messages.append(AIMessage(content=error_message))

elif not google_api_key:
    st.warning("⚠️ Silakan masukkan Google API Key Anda di sidebar terlebih dahulu.")
elif not uploaded_file:
    st.info("📤 Silakan unggah dokumen PDF untuk memulai.")