import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')


def load_and_split_data(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # print(chunks)
    return chunks


def embed(file_path):
    chunks = load_and_split_data(file_path)
    vdb = get_vector_db()
    vdb.add_documents(chunks)
    # vdb.persist()


def get_vector_db():
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL, show_progress=True)

    vdb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    return vdb





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
    # document = chunks[0]
    # print(document.page_content)
    # print(document.metadata)

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

    print("*" * 100)
    # Create a new Chroma database from the documents using OpenAI embeddings
    # embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    embedding_function = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)
    db = Chroma.from_documents(documents=chunks,
                               embedding=embedding_function,
                               persist_directory=CHROMA_PATH)

    print(db)

    results = db.similarity_search("artificial intelligence", k=2)
    print(results)
    print("*" * 100)
    # Persist the database to disk
    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store():
    """
    Function to generate vector database in chroma from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_chroma(chunks)  # Save the processed data to a data store


def test_vector_db(qns):
    # embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    embedding_function = OllamaEmbeddings(model='nomic-embed-text', show_progress=True)
    vector_store = Chroma(embedding_function=embedding_function,
                          persist_directory=CHROMA_PATH)

    results = vector_store.similarity_search("future of ai", k=2)
    print(results)

    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # documents = db.similarity_search_with_relevance_scores(qns, k=3)

    retriever = vector_store.as_retriever()
    documents = retriever.invoke(qns)
    print(documents)


if __name__ == "__main__":
    embed("./upload/CDOS_Administration_Guide.pdf")
    # db = get_vector_db()
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 20,
    #         "score_threshold": 0.1,
    #     },)
    # documents = retriever.invoke("adding study")
    # print(documents)
