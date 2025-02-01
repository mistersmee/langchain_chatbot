from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, render_template  # Add render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class BrainloxCourseLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> List[Document]:
        # Fetch the webpage
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            html_content = response.text
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return []

        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all course content divs
        course_divs = soup.find_all('div', class_='courses-content')

        documents = []
        for div in course_divs:
            # Extract course information
            title_elem = div.find('h3')
            title = title_elem.text.strip() if title_elem else ''

            desc_elem = div.find('p')
            description = desc_elem.text.strip() if desc_elem else ''

            lessons_elem = div.find('i', class_='flaticon-agenda')
            lessons = ''
            if lessons_elem and lessons_elem.parent:
                lessons = lessons_elem.parent.text.strip()

            # Create a structured text representation
            content = f"""
            Course: {title}
            Description: {description}
            Lessons: {lessons}
            """

            # Create metadata
            metadata = {
                'source': self.url,
                'title': title,
                'type': 'course'
            }

            # Create Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

def setup_rag_pipeline():
    # Initialize the loader
    loader = BrainloxCourseLoader("https://brainlox.com/courses/category/technical")

    # Load documents
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./data/chroma_db"
    )

    import os

    # Get HuggingFace API token from environment variable
    huggingface_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if not huggingface_token:
        raise ValueError("Please set HUGGINGFACE_API_TOKEN environment variable")

  # Setup TinyLlama through HuggingFace
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        huggingfacehub_api_token=huggingface_token,
        task="text-generation",
    )

    # Create prompt template
    prompt = PromptTemplate(
        template="""Use the following context to answer the question.

        Context: {context}

        Question: {question}

        Answer in complete sentences.""",
        input_variables=["context", "question"]
    )

    # Create and return the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def query_courses(qa_chain, query: str) -> str:
    try:
        # Simply use the chain's invoke method
        response = qa_chain.invoke({"query": query})
        return response["result"]
    except Exception as e:
        print(f"Error in query_courses: {e}")
        return str(e)

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Example query
        qa_chain = setup_rag_pipeline()
        answer = query_courses(qa_chain, user_message)

        return jsonify({"response": answer}), 200

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Webpage endpoint
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML page

if __name__ == '__main__':
    app.run(debug=True)
