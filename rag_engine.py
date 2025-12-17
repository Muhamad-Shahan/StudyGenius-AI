import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. CONFIGURATION
# We set the model here so we can change it easily later
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def get_llm(hf_token):
    """Sets up the Language Model"""
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    return ChatHuggingFace(llm=llm)

def process_document(uploaded_file):
    """
    Takes a raw file upload, saves it temporarily,
    loads it, splits it, and returns the chunks.
    """
    # Save the file locally so PyPDFLoader can read it
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and Split
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Clean up (delete temp file)
    os.remove(temp_path)
    return chunks

def create_vectorstore(chunks):
    """Turns text chunks into a searchable FAISS database"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_chat_chain(vectorstore, llm):
    """Creates the Q&A Chain"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    <|system|>
    You are a helpful study assistant. Answer the question based strictly on the context below.
    If the answer is not in the context, say "I don't know" and do not make it up.
    Keep answers concise (max 3 sentences).
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

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_quiz_chain(vectorstore, llm):
    """Creates a special chain just for generating Quizzes"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Get more context for a quiz

    quiz_template = """
    <|system|>
    You are a strict Professor. Based on the context provided, generate a quiz.
    - Create 3 Multiple Choice Questions.
    - Provide 4 options (A, B, C, D) for each.
    - Indicate the correct answer at the very end.
    </s>
    <|user|>
    Context: {context}
    Topic: Generate a quiz based on this text.
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(quiz_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
