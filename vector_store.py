from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic import NomicEmbeddings

CHROMA_PATH = "./db_index"  # Path to the directory to save Chroma database
DATA_PATH = "./upload"  # Directory to your pdf files


def embed_documents(f_path):
    # file_path = "./upload/CDOS_Administration_Guide.pdf"
    # file_path = "./upload/ED604401.pdf"
    loader = PyPDFLoader(f_path)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        persist_path=CHROMA_PATH,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        serializer="parquet",
    )

    vectorstore.persist()
    print("Vector store was persisted to", CHROMA_PATH)


def get_retriever():
    vectorstore = SKLearnVectorStore(
        persist_path=CHROMA_PATH,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
        serializer="parquet"
    )

    # Create retriever
    # retriever = vectorstore.as_retriever(k=3)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1,
        },)
    return retriever


if __name__ == "__main__":
    # file_path = "./upload/ED604401.pdf"
    file_path = "./upload/CDOS_Administration_Guide.pdf"
    embed_documents(file_path)
