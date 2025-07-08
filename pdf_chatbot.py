from typing import List
import os

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory

class PDFChatBot:
    def __init__(self, api_key: str):
        """
        Inisialisasi ChatBot dengan Google API Key
        """
        self.api_key = api_key
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def process_pdf(self, pdf_path: str) -> None:
        """
        Memproses file PDF dan membuat vector store
        
        1. Load PDF
        2. Split menjadi chunks
        3. Convert ke embeddings
        4. Simpan dalam vector store
        """
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split dokumen menjadi chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Ukuran setiap chunk
            chunk_overlap=200, # Overlap antar chunk untuk konteks
            length_function=len
        )
        documents = text_splitter.split_documents(pages)

        # Buat embeddings menggunakan model Google
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )

        # Buat vector store
        self.vector_store = FAISS.from_documents(documents, embeddings)
        
        # Buat chain untuk QA
        self._create_chain()

    def _create_chain(self) -> None:
        """
        Membuat chain untuk question-answering
        
        1. Inisialisasi LLM
        2. Buat prompt template
        3. Buat document chain
        4. Buat retrieval chain
        """
        # Inisialisasi LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )

        # Buat prompt template
        prompt = ChatPromptTemplate.from_template("""
        Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
        
        Riwayat Chat:
        {chat_history}
        
        Gunakan konteks berikut untuk menjawab pertanyaan:
        {context}
        
        Pertanyaan: {input}
        
        Berikan jawaban yang jelas dan informatif. Jika Anda tidak dapat menjawab berdasarkan konteks yang diberikan,
        katakan bahwa Anda tidak dapat menemukan informasi yang relevan dalam dokumen.
        """)

        # Buat document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Buat retrieval chain
        self.chain = create_retrieval_chain(
            self.vector_store.as_retriever(search_kwargs={"k": 3}),
            document_chain
        )

    def get_answer(self, question: str) -> dict:
        """
        Mendapatkan jawaban untuk pertanyaan
        
        1. Ambil chat history dari memory
        2. Jalankan chain dengan input dan history
        3. Simpan interaksi ke memory
        4. Return jawaban
        """
        if not self.chain:
            raise ValueError("Please process a PDF first using process_pdf()")

        # Ambil chat history
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])

        # Jalankan chain
        response = self.chain.invoke({
            "input": question,
            "chat_history": chat_history
        })

        # Simpan ke memory
        self.memory.save_context(
            {"input": question},
            {"answer": response["answer"]}
        )

        return response

def main():
    """
    Contoh penggunaan PDFChatBot
    """
    # Inisialisasi bot
    api_key = "YOUR_GOOGLE_API_KEY"  # Ganti dengan API key Anda
    bot = PDFChatBot(api_key)

    # Proses PDF
    pdf_path = "path/to/your/document.pdf"  # Ganti dengan path PDF Anda
    bot.process_pdf(pdf_path)

    # Contoh interaksi
    questions = [
        "Apa topik utama dari dokumen ini?",
        "Bisakah Anda jelaskan lebih detail tentang bagian pertama?",
        "Apa kesimpulan dari dokumen ini?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        response = bot.get_answer(question)
        print(f"A: {response['answer']}")

if __name__ == "__main__":
    main() 