import os
from dotenv import load_dotenv

from fastapi import FastAPI

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import pinecone

# uvicorn chatai-service:app --reload
app = FastAPI(title="aichatbot-backend-service", description="Our backend services provide a robust and scalable solution, \
              enabling seamless integration of conversational AI capabilities into your system. By leveraging the extensive \
              language understanding and generation capabilities of GPT-3.5, our chatbot backend services can comprehend and \
              respond to user queries with remarkable accuracy, context awareness, and natural language understanding.")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV,  # next to api key in console
)

index = pinecone.Index("pi-test")

index.upsert([
    ("F", [0.1, 0.1, 0.1, 0.1, 0.1]),
    ("G", [0.2, 0.2, 0.2, 0.2, 0.2])
],namespace='my-first-namespace')