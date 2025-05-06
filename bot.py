import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Set environment variables for LangChain and other services
        self.LANGSMITH_TRACING = "true"
        self.LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
        self.LANGSMITH_API_KEY = os.getenv('langsmith_api_key')
        self.LANGSMITH_PROJECT = "pdf-rag-model"
        self.GOOGLE_API_KEY = os.getenv('google_api_key')
        self.GROQ_API_KEY = os.getenv('groq_api_key')

        # Set environment variables for LangChain
        os.environ["LANGCHAIN_TRACING_V2"] = self.LANGSMITH_TRACING
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT

        # Set Google API Key
        os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY

        # Initialize model and embeddings
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.model = ChatGroq(groq_api_key=self.GROQ_API_KEY, model="gemma2-9b-it")

        # Load and split documents
        self.loader = TextLoader('book.txt', encoding='utf-8')
        self.documents = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        self.docs = self.text_splitter.split_documents(self.documents)

        # Create the Chroma vector store
        self.vectorstore = Chroma.from_documents(documents=self.docs, embedding=self.embedding)

        # Create the retriever for the vectorstore
        self.retriever = self.vectorstore.as_retriever()

        # Define the system and chat prompts
        self.system_prompt = (
            "You are an assistant for question answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

        # Create the chat prompt template
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )

        # Create the chain for question answering
        self.question_answering_chain = create_stuff_documents_chain(self.model, self.chat_prompt)

        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answering_chain)

    def answer_question(self, input_text):
        """
        Given an input question, retrieve relevant documents and generate an answer.
        """
        result = self.rag_chain.invoke({"input": input_text})
        return result['answer']

# # Example usage:
# if __name__ == "__main__":
#     bot = ChatBot()
#     input_question = "What is Chapter 9 about?"
#     answer = bot.answer_question(input_question)
#     print(answer)
