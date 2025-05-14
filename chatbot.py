import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

GOOGLE_API_KEY = "API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


st.header("Introduction to the Gen AI")



#Upload the file
with st.sidebar:
    st.subheader("Upload Documents")
    file = st.file_uploader("Upload a pdf file and start asking questions")

if file is not None:
    #Extract the file content
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    #Split the file content into chunks
    text_splitter = RecursiveCharacterTextSplitter(separators="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)


    #Generate embeddings for the pdf file data
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    #Store it in Vector Store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    #Take user question as input
    user_question = st.text_input("Enter your question")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)

        # define the llm
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
