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
import hashlib
import os

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

def document_hash(doc: Document) -> str:
    """
    Compute a hash of the document content to identify duplicates.
    """
    return hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()


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

    # Load or create the vector store
    persist_dir = "./data/chroma_db"
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,  # Explicitly pass the embedding function
        )
    except Exception:
        # If the vector store doesn't exist, initialize it with an empty collection
        vectorstore = Chroma.from_documents([], embedding=embeddings, persist_directory=persist_dir)

 # Retrieve existing hashes from metadata
    existing_hashes = set()
    try:
        existing_docs = vectorstore.similarity_search("", k=1000)  # Retrieve all docs
        for doc in existing_docs:
            hash_value = doc.metadata.get("hash", None)
            if hash_value:
                existing_hashes.add(hash_value)
    except Exception as e:
        print(f"Error retrieving existing documents: {e}")

    # Filter out duplicates
    new_documents = []
    for doc in splits:
        doc_hash = document_hash(doc)
        if doc_hash not in existing_hashes:
            doc.metadata["hash"] = doc_hash  # Add hash to metadata
            new_documents.append(doc)

    if new_documents:
        # Add new documents to the vector store
        vectorstore.add_documents(new_documents)
        vectorstore.persist()
        print(f"Added {len(new_documents)} new documents to the vector store.")
    else:
        print("No new documents to add to the vector store.")

    # Get HuggingFace API token from environment variable
    huggingface_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if not huggingface_token:
        raise ValueError("Please set HUGGINGFACE_API_TOKEN environment variable")

  # Setup TinyLlama through HuggingFace
    llm = HuggingFaceHub(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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
    """
    Query the RAG pipeline and extract the actual answer from the response.
    """
    # Get the raw response from the chain
    raw_response = qa_chain.run(query)

    # Define the marker used in the prompt
    answer_marker = "Answer in complete sentences."

    # Extract the answer after the marker
    if answer_marker in raw_response:
        actual_answer = raw_response.split(answer_marker, 1)[1].strip()  # Take everything after the marker
    else:
        actual_answer = raw_response  # Fallback in case the marker is not found

    return actual_answer

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
