import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-proj-"  # Replace with your actual API key

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User model for login
class User(BaseModel):
    username: str
    password: str

# Query model for chat
class Query(BaseModel):
    message: str

# Dummy user data for authentication (replace in production)
DUMMY_USER = {"username": "testuser", "password": "password123"}

# Helper: Load text from file
def load_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"File not found: {file_path}")

# Helper: Split text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Setup function for Retrieval-Augmented Generation (RAG)
def setup_rag_application(file_path: str):
    try:
        print("Loading file content...")
        text_content = load_text_file(file_path)
        print("Splitting text into chunks...")
        text_chunks = split_text_into_chunks(text_content)
        print(f"Total chunks created: {len(text_chunks)}")

        print("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        print("Creating vector store...")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        print("Setting up retriever and LLM...")
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        print("Initializing RetrievalQA chain...")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        return qa_chain
    except Exception as e:
        print(f"Error during RAG setup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during RAG setup: {str(e)}")

# Initialize QA Chain
file_path = "output.txt"
try:
    qa_chain = setup_rag_application(file_path)
    print("RAG setup complete and QA chain initialized.")
except HTTPException as e:
    print(f"RAG setup error: {e.detail}")
    qa_chain = None

# Login endpoint
@app.post("/login")
async def login(user: User):
    if user.username == DUMMY_USER["username"] and user.password == DUMMY_USER["password"]:
        return {"access_token": "dummy-token"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Chat endpoint
@app.post("/chat")
async def chat(query: Query, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")

    token = authorization[len("Bearer "):]
    if token != "dummy-token":
        raise HTTPException(status_code=401, detail="Invalid token")

    if not qa_chain:
        raise HTTPException(status_code=500, detail="System not initialized properly. Please try again later.")

    try:
        print(f"Received query: {query.message}")
        response = qa_chain.run(query.message)
        print(f"Response from QA chain: {response}")
        if not response:
            response = "No response available."
        return {"response": response}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if 'insufficient_quota' in str(e):
            raise HTTPException(status_code=429, detail="Quota exceeded. Please check your OpenAI billing and usage.")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
