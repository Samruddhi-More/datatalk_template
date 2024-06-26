import os
import requests
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import PyPDF2
from PyPDF2 import PdfReader
from docx2pdf import convert
import pandas as pd
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


class DataTalk:

    def __init__(self, url):
        self.url = url
        self.filepath = self.download_file(url)

    def download_file(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            # Extract the filename from the URL and save it in the current directory
            filename = url.split("/")[-1].split("?")[0]  # Remove query parameters if any
            with open(filename, "wb") as file:
                file.write(response.content)
            return filename
        else:
            raise Exception(f"Failed to download file from {url}")

    def source_type(self):
        file = ''
        folderpath = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)
        if filename.endswith('.xlsx'):
            csvname = filename.split('.')[0] + '.csv'
            file = os.path.join(folderpath, csvname)
            df = pd.read_excel(self.filepath)
            df.to_csv(file)
        elif filename.endswith('.docx'):
            pdfname = filename.split('.')[0] + '.pdf'
            file = os.path.join(folderpath, pdfname)
            convert(self.filepath, file)
        else: 
            file = self.filepath

        return file

    def generate_response(self, query):
        # Initialize the OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        model_name = "gpt-3.5-turbo-instruct"
        file = self.source_type()
        response = ''

        if file.endswith('.csv'):
            # Load the documents
            loader = CSVLoader(file_path=file)
            # Create an index using the loaded documents
            index_creator = VectorstoreIndexCreator(embedding=embeddings)
            docsearch = index_creator.from_loaders([loader])
            # Create a question-answering chain using the index
            llm = OpenAI(model_name=model_name, openai_api_key=openai.api_key)
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
            response = chain({"question": query})
            response = response['result']

        elif file.lower().endswith('.pdf'):
            """
            Process a PDF file, extracting text and splitting it into chunks.
            """
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # Create vector store using FAISS
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)

            """
            Generate responses based on prompts and embeddings.
            """
            retriever = vector_store.as_retriever()
            
            # Define template for prompts
            template = """Respond to the prompt based on the following context: {context}
            Questions: {questions}
            """
            prompt = ChatPromptTemplate.from_template(template)

            # Initialize ChatOpenAI model
            model = ChatOpenAI(openai_api_key=openai.api_key)
            
            # Define processing chain
            chain = (
                {"context": retriever, "questions": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )
            
            # Invoke the processing chain with user input
            response = chain.invoke(query)
        
        else:
            print('The data source format is not in required format')

        return response

# # Example usage
# chat = DataTalk('https://drive.google.com/uc?id=1zO8ekHWx9U7mrbx_0Hoxxu6od7uxJqWw&export=download')
# response = chat.generate_response('What is the doc about?')
# print(response)