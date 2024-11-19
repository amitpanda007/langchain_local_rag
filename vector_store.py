import os
import shutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

# https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339


CHROMA_PATH = "chroma"  # Path to the directory to save Chroma database
DATA_PATH = "/upload/"  # Directory to your pdf files


def get_retriever():
    # file_path = "./upload/CDOS_Administration_Guide.pdf"
    file_path = "./upload/ED604401.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)

    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)
    return retriever


def load_documents():
    """
    Load PDF documents from the specified directory using PyPDFDirectoryLoader.
    Returns:
    List of Document objects: Loaded PDF documents represented as Langchain
                                                            Document objects.
    """
    # Initialize PDF loader with specified directory
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    # Load PDF documents and return them as a list of Document objects
    return document_loader.load()


def split_text(documents: list[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
      documents (list[Document]): List of Document objects containing text content to split.
    Returns:
      list[Document]: List of Document objects representing the split text chunks.
    """
    # Initialize text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print example of page content and metadata for a chunk
    document = chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Save the given list of Document objects to a Chroma database.
    Args:
    chunks (list[Document]): List of Document objects representing text chunks to save.
    Returns:
    None
    """

    # Clear out the existing database directory if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the documents using OpenAI embeddings
    db = Chroma.from_documents(
        chunks,
        NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        persist_directory=CHROMA_PATH
    )

    # Persist the database to disk
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store


