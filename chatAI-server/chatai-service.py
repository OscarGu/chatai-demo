"""
Filename: chatai-service.py
Author: Oscar Gu
Contact: gudanyang@live.cn
"""

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

embeddings = OpenAIEmbeddings()

# ws
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# filepath: https://www-file.huawei.com/-/media/CORPORATE/PDF/public-policy/public_policy_position_5g_spectrum.pdf
# upload and embbeding file into VectorStoreDB


'''
    - param filepath: url format filepath
    - param index_name: Pinecone index
    - param namespace: namespace: Pinecone namespace with the specific index
'''
@app.post("/embedFile/", tags=["VectorDB(Pinecone) Operations"], summary=("Embed file into vector DB"),
          description="upload (pdf)document, split the document into chunks and embed it into vector database(pinecone).")
def embedFile(filepath: str, index_name: str, namespace: str):
    """Return the pathname of the KOS root directory."""
    doc = get_file_loader(filepath)
    docs = getSplitDocs(doc)
    docsearch = Pinecone.from_documents(
        documents=docs, embedding=embeddings, index_name=index_name, namespace=namespace)
    return {
        "Success": {
            "index_name": index_name,
            "namespace": namespace
        }
    }


# ws - remove index from vector databases
@app.delete("/delete-namespace/{index_name}/{namespace}", tags=["VectorDB(Pinecone) Operations"])
def delete_index(index_name: str, namespace: str):
    """
        Remove content by namespace in an index
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV,  # next to api key in console
    )
    pinecone.Index(index_name).delete(delete_all=True, namespace=namespace)
    return {"Success": "namespace is deleted successfully."}


@app.post("/chatWithDoc", tags=["Chatbot Operations"])
def chatWithDoc(index_name: str, namespace: str, query: str):
    docsearch = load_existing_docs(index_name=index_name, namespace=namespace)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(),
        memory=memory,
        # verbose=True
    )
    chat_history=[]
    result = conversation_chain({"question": query, "chat_history": chat_history})
    # result = conversation_chain({"question": query})
    return result

# load existing index from Pinecone


def load_existing_docs(index_name: str, namespace: str):
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV,  # next to api key in console
    )
    docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace)
    return docsearch


# retrieve info from existing index with namespace


def getAIChatbot(index_name: str, namespace: str):
    '''
        return: conversation_chain
    '''
    docsearch = load_existing_docs(index_name=index_name, namespace=namespace)
    chat_llm = OpenAI(temperature=0)

    chat_buffer_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=docsearch.as_retriever(),
        memory=chat_buffer_memory,
        # verbose=True
    )
    return conversation_chain

# get chat history per user id


def getChatHistory(userID):
    pass


def talkWithAIChatbot(index_name, namespace, useuserID):
    bot = getAIChatbot(index_name, namespace)


def get_file_loader(filepath: str):
    print(f'get file from {filepath}')
    loader = PyPDFLoader(filepath)
    document = loader.load()
    print(f'You have {len(document)} document(s) in your data')
    return document


def getSplitDocs(document: list, chunk_size = 1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    print(
        f'begin to use splitter with chunk_size:{chunk_size}, chunk_overlap:{chunk_overlap}, to split the file.')
    docs = text_splitter.split_documents(document)
    print(f'Now you have {len(docs)} split documents')
    return docs


@app.get("/", tags=["Server Info"])
def index():
    return {"Server Info": "The services are running successfully.",
            "Doc Info": "Please go to serverurl/docs to visit the help documents."}
