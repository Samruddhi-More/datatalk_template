from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.llms import OpenAI
from dotenv import load_dotenv
import openai
import os
import pandas as pd
from docx2pdf import convert

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(temperature=0,model_name="llama-3.1-8b-instant", api_key=api_key)

class DataTalk:
    _instance = None
    _initialized = False

    def __new__(cls, **args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataTalk, cls).__new__(cls)
        return cls._instance
    
    # def source_type(self, filepath):
    #     # file = ''
    #     folderpath = os.path.dirname(self.filepath)
    #     filename = os.path.basename(self.filepath)
    #     if filename.endswith('.xlsx'):
    #         csvname = filename.split('.')[0] + '.csv'
    #         file = os.path.join(folderpath, csvname)
    #         df = pd.read_excel(self.filepath)
    #         df.to_csv(file)
    #     elif filename.endswith('.docx'):
    #         pdfname = filename.split('.')[0] + '.pdf'
    #         file = os.path.join(folderpath, pdfname)
    #         convert(self.filepath, file)
    #     else: 
    #         file = self.filepath

    #     return file
    
    def data_loader(self, filepath):
        loader = None
        if filepath.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(filepath)
        if filepath.endswith('.csv'):
            loader = CSVLoader(file_path=filepath)
        if filepath.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        if filepath.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(filepath)
        if 'http' in filepath:
            loader = UnstructuredURLLoader(urls=[filepath])
        if loader is not None:
            data = loader.load()
            return data
        else:
            raise ValueError("Unsupported file type or URL")

    
    def __init__(self, path):
        if not self._initialized:
            self.path = path

            self.data = self.data_loader(self.path)

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=10,
                length_function=len
            )

            self.chunks = self.text_splitter.split_documents(self.data)
            self.embeddings = HuggingFaceBgeEmbeddings()
            self.vector_store = Chroma.from_documents(
                self.chunks,
                # collection_name="example_collection",
                self.embeddings,
                persist_directory="./datatalk_db",  # Where to save data locally, remove if not neccesary
            )
            self.retriever = self.vector_store.as_retriever()

            self.template = """Respond to the user query:{query} based on the following context: {context}
            """
            self.prompt = ChatPromptTemplate.from_template(self.template)

            self.chain =  self.prompt | llm

            self._initialized = True
    
    def get_response(self,query):
        if self._initialized:
            response = self.chain.invoke({"query": query,"context": self.retriever.get_relevant_documents(query)})

            return response





# class DataTalk:

#     def __init__(self):
#         urls = ["https://docs.socketmobile.com/main/en/?_gl=1*1gbx00q*_gcl_au*MTUzNjYyNTExNC4xNzIxMDU2ODc3*_ga*OTMzMzM4Mzc0LjE3MjEwNTY4Nzc.*_ga_98RX67473S*MTcyMTMyODYxNi44LjEuMTcyMTMyOTU4My41NS4wLjA"]
#         loaders = UnstructuredURLLoader(urls=urls)
#         data = loaders.load()
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text=str(data))
#          # Create vector store using FAISS
#  # Initialize the OpenAI embeddings
#         embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
#         vector_store = FAISS.from_texts(chunks, embedding=embeddings)

#         self.retriever = vector_store.as_retriever()

#         # Define template for prompts
#         template = """
#         You are a Customer Service Advisor. Your task is to use the provided context as a standard code document and perform the following tasks:
#         - Assist the user in installing the software based on the guidance mentioned in the context.
#         - Resolve any user queries related to the software.

#         The following information is given to you:

#         Context:
#         {context}

#         Questions:
#         {questions}
#         """
#         self.prompt = ChatPromptTemplate.from_template(template)

#         # Create a question-answering chain using the index
#         self.llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai.api_key)
    

#     def generate_response(self, query):
#         """
#             Generate responses based on prompts and embeddings.
#             """
#         # Define processing chain
#         chain = (
#             {"context": self.retriever, "questions": RunnablePassthrough()}
#             | self.prompt
#             | self.llm
#             | StrOutputParser()
#         )

#         # Invoke the processing chain with user input
#         response = chain.invoke(query)

#         return response

