import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Set page configuration
st.set_page_config(
    page_title="Simple Chatbot with Memory 🤖",
    page_icon="🤖",
    layout="wide"
)

# Initialize session states
def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Halo! Saya adalah chatbot yang bisa mengingat konteks percakapan. Apa yang ingin Anda bicarakan?")
        ]
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

# Call initialization
initialize_session_state()

# Display the title
st.title("Simple Chatbot with Memory 🤖")

# Sidebar for configuration
with st.sidebar:
    st.header("Konfigurasi")
    google_api_key = "AIzaSyAf12tw3KOkJH-DxPAFickgHmI5pU8I9sI"
    
    # Add separator
    st.markdown("---")
    
    # Chat History Section
    st.header("Riwayat Chat 💬")
    
    # Add clear chat button
    if st.button("Bersihkan Riwayat Chat 🗑️"):
        st.session_state.messages = [
            AIMessage(content="Halo! Saya adalah chatbot yang bisa mengingat konteks percakapan. Apa yang ingin Anda bicarakan?")
        ]
        st.session_state.memory.clear()
        st.session_state.conversation = None
        st.rerun()
    
    # Display chat history
    if len(st.session_state.messages) > 1:
        st.markdown("#### Pesan Terakhir")
        for idx, msg in enumerate(st.session_state.messages[-5:]):
            if isinstance(msg, HumanMessage):
                with st.container():
                    st.markdown("**🧑 Anda:**")
                    st.info(msg.content)
            elif isinstance(msg, AIMessage):
                with st.container():
                    st.markdown("**🤖 AI:**")
                    st.success(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
            
            if idx < len(st.session_state.messages[-5:]) - 1:
                st.markdown("---")
    else:
        st.info("Belum ada riwayat chat. Mulai dengan mengajukan pertanyaan!")

    with st.expander("ℹ️ Tentang Chatbot"):
        st.markdown("""
        - Menggunakan ConversationBufferMemory
        - Mengingat konteks percakapan sebelumnya
        - Bisa memberikan jawaban berdasarkan percakapan sebelumnya
        - Model: Google Gemini
        """)

def initialize_conversation():
    """Initialize the conversation chain with memory"""
    # Initialize the language model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.7
    )

    # Create a custom prompt template
    prompt_template = """Anda adalah asisten AI yang ramah dan membantu. Anda memiliki kemampuan untuk mengingat percakapan sebelumnya dan memberikan jawaban yang kontekstual.

Riwayat Percakapan:
{history}

Manusia: {input}
AI: """

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template
    )

    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=prompt,
        verbose=True
    )

    return conversation

# Initialize conversation if not exists
if st.session_state.conversation is None:
    st.session_state.conversation = initialize_conversation()

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            # Get response from conversation chain
            response = st.session_state.conversation.predict(input=prompt)
            
            # Display response
            response_placeholder.write(response)
            
            # Add AI response to chat history
            st.session_state.messages.append(AIMessage(content=response))
            
        except Exception as e:
            error_message = f"Maaf, terjadi kesalahan: {str(e)}"
            response_placeholder.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message)) 