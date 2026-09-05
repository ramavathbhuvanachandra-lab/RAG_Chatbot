from langchain_chroma import Chroma
from backend.embedding import embeddings

vectorstore = Chroma(
    collection_name="iitj_v1",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

