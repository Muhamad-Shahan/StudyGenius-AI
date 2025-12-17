import os
import time
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# FALLBACK MODELS
# If Zephyr fails, we can swap this variable to "mistralai/Mistral-7B-Instruct-v0.3"
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def get_llm(hf_token):
    """Sets up the Language Model with error handling"""
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token.strip() # .strip() removes accidental spaces!
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=REPO_ID,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            temperature=0.1
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        # This prints the REAL error to your console so you can see it
        print(f"❌ Error initializing model: {e}")
        raise e

def process_document(uploaded_file):
    """Takes a raw file upload, saves it temporarily, and splits it."""
    try:
        # Save the file locally
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and Split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Larger chunks for better context
        chunks = splitter.split_documents(docs)

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return chunks
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

def create_vectorstore(chunks):
    """Turns text chunks into a searchable FAISS database"""
    if not chunks:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None

def get_chat_chain(vectorstore, llm):
    """Creates the Q&A Chain"""
    # Safety check
    if vectorstore is None or llm is None:
        raise ValueError("Vectorstore or LLM is missing.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    <|system|>
    You are a helpful study assistant. Answer the question based strictly on the context below.
    If the answer is not in the context, say "I don't know".
    Keep answers concise.
    </s>
    <|user|>
    Context: {context}
    Question: {question}
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def get_quiz_chain(vectorstore, llm):
    """Creates the Quiz Chain"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    quiz_template = """
    <|system|>
    You are a Professor. Based on the context provided, generate a 3-question Multiple Choice Quiz.
    Format:
    Q1: [Question]
    A) [Option]
    B) [Option]
    C) [Option]
    D) [Option]
    Correct Answer: [Option]
    
    (Repeat for Q2 and Q3)
    </s>
    <|user|>
    Context: {context}
    Topic: Generate a quiz.
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(quiz_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )
