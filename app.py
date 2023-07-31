import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.llms import AzureOpenAI,OpenAI
import os
from langchain.document_loaders import DirectoryLoader
from PIL import Image
from langchain.document_loaders import UnstructuredExcelLoader
from openpyxl import reader,load_workbook,Workbook
import pandas as pd
import io
from langchain.document_loaders import UnstructuredFileLoader
import docx
import pinecone
from langchain.vectorstores import Pinecone
#--Below code is replaced with Directory Loader....
def get_pdf_text(pdf_docs):
    text  = ""
    text1 = ""
    text2 = ""
    fullText=[]
    for pdf in pdf_docs:
        if 'pdf' in pdf.name:
            #print(pdf.name)
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif 'txt'in pdf.name:
            text1 = pdf.read()
            #print(text1)
        elif 'xls'in pdf.name:
            df = pd.read_excel(pdf)
            #print(df)
            text2 = df.to_string()
        elif 'docx'in pdf.name:
            print('Docx')
            text3 = docx.Document(pdf)
            print(text3)
            for para in text3.paragraphs:
                fullText.append(para.text)
            print('\n'.join(fullText))
            st.write(fullText)
            #loader = UnstructuredFileLoader(pdf)
            #text3 = loader.load()
            print(fullText)
            #text2 = df.to_string()    
    return text + str(text1)  + text2 + (str(fullText).strip('[]'))
#----------------------------------------------------

#--Load all files of the directory
directory = 'c:/Temp/Data/'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  #print(str(documents).strip('[]'))
  documents = (str(documents).strip('[]'))
  documents = documents.replace("\n", " ")
  print(documents)
  return documents
#----------------------------------------------------


#--Create chunks of the documents loaded 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
#--------------------------------------------------

# Embedding and Storing the info to Vector DB ------
def get_vectorstore(text_chunks):
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://virtualanalyticsassistant.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = 'b1cf2fd35ec64c9b8247e0feac960f9c'
    embeddings = OpenAIEmbeddings(deployment="Embedding-ADA-002", chunk_size = 1)
    #embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # initialize pinecone

    # Vector DB Configuration
    
    # 1. -- Pinecone Integration for Vector DB
    #pinecone.init(
    #    api_key="b2d6c698-7e6f-4ae0-ae54-43e74a9fc22e",  # find at app.pinecone.io
    #    environment="us-west4-gcp-free"  # next to api key in console
    #)
    #index_name = "newpinecone"
    #vectorstore = pinecone.Index(index_name)
    #st.write(vectorstore)
    #results = index.query(queries=query_vectors)
    #vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    #vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    #Pinecone.from_existing_indexes(texts=text_chunks, embedding=embeddings)
    #2. FAISS Integration for Vector DB
    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore
#--------------------------------------------------

def get_vectorstore_existing():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://virtualanalyticsassistant.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = 'b1cf2fd35ec64c9b8247e0feac960f9c'
    embeddings = OpenAIEmbeddings(deployment="Embedding-ADA-002", chunk_size = 2)
    # initialize pinecone

    # Vector DB Configuration
    
    # 1. -- Pinecone Integration for Vector DB
    pinecone.init(
        api_key="b2d6c698-7e6f-4ae0-ae54-43e74a9fc22e",  # find at app.pinecone.io
        environment="us-west4-gcp-free"  # next to api key in console
    )
    index_name = "newpinecone"
    #vectorstore = pinecone.Index(index_name)
    #st.write(vectorstore)
    #results = index.query(queries=query_vectors)
    #vectorstore = Pinecone.from_texts(texts=text_chunks, embedding=embeddings, index_name=index_name)
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    #Pinecone.from_existing_indexes(texts=text_chunks, embedding=embeddings)
    #2. FAISS Integration for Vector DB
    
    #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore
#--------------------------------------------------
# LLM usage and buffering the results ------
def get_conversation_chain(vectorstore):
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://virtualanalyticsassistant.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = 'b1cf2fd35ec64c9b8247e0feac960f9c'
    #os.environ["OPENAI_API_KEY"] = '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'
    llm = AzureOpenAI(deployment_name="GPT-Turbo-35", model_name="gpt-35-turbo")
    #llm = OpenAI(model_name='gpt-4',openai_api_key='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')
    #llm = ChatOpenAI(engine='text-davinci-003',openai_api_key='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',)
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    print('getting answer')
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
#--------------------------------------------------

# User Prompt Inputs ------
def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        #st.write(response)
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except:
        e = RuntimeError('This is an exception of type RuntimeError')
        st.exception(e)
#--------------------------------------------------

def main():
    load_dotenv()
    im = Image.open('Header.jpg')
    st.set_page_config(page_title="Endeavor Filebot", page_icon = im)
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    #image = Image.open('endeavorlogo.jpg')
    #st.image(image, caption='')
    st.header("Chat with Your Documents")
    
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)


    with st.sidebar:
        image = Image.open('endeavorlogo.jpg')
        st.image(image, caption='')
            # Below File Uploaded functionality commented..using only Process button to process all the documents of the folder
        
        st.subheader("Upload Your documents")
        load_docs = st.file_uploader("Upload your Documents here and click on 'Process'", accept_multiple_files=True)
        
        #-------------------------

        if st.button("Process Selected Documents"):
            with st.spinner("Processing"):
                if load_docs:
                    print('Not Empty')
                    print(load_docs)
                    # get pdf text
                    raw_text = get_pdf_text(load_docs)
                    #raw_text = load_docs(directory)
                    # get the text chunks
                    print('Reading Done')
                    #print(raw_text)

                    text_chunks = get_text_chunks(raw_text)
                    print('Chunks Created')
                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    #st.write('aaa')
                    #vectorstore = get_vectorstore_existing()
                    print('Vector DB Store')
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    print('Sessions Created')
                    
                else:
                    print('Empty')
                    print(load_docs)
                    vectorstore = get_vectorstore_existing()
                    print('Vector DB Store')
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    print('Sessions Created')

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        with st.spinner("Retrieving Answer"):
            handle_userinput(user_question)
         
if __name__ == '__main__':
    main()