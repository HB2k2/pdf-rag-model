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
from langchain_core.prompts import MessagesPlaceholder

class ChatBot():
    load_dotenv()

    LANGSMITH_TRACING="true"
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    LANGSMITH_API_KEY=os.getenv('langsmith_api_key')
    LANGSMITH_PROJECT="pdf-rag-model"
    GOOGLE_API_KEY=os.getenv('google_api_key')
    GROQ_API_KEY=os.getenv('groq_api_key')

    os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"]= LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]= LANGSMITH_PROJECT

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model = ChatGroq(groq_api_key=GROQ_API_KEY, model="gemma2-9b-it")

    loader = TextLoader('book.txt', encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    vectorestore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
    )

    retriever = vectorestore.as_retriever()

    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answering_chain = create_stuff_documents_chain(model, chat_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    

# bot = ChatBot()
# input = "What is Chapter 9 about?"
# result = bot.rag_chain.invoke({"input": input})
# print(result['answer'])

