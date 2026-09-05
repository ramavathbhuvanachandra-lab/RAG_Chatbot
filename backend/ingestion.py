import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from backend.embedding import embeddings


def load_documents(data_path: str):
    """
    Load all .docx documents recursively from the given directory.
    """

    loader = DirectoryLoader(
        data_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    )

    documents = loader.load()

    return documents


def split_documents(documents):
    """
    Split loaded documents into chunks for embedding.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vectorstore(chunks):
    """
    Delete the existing Chroma database and create a fresh vector database
    by embedding and inserting documents in smaller batches.
    """

    persist_directory = "./chroma_db"

    if os.path.exists(persist_directory):
        print("Deleting existing Chroma database...")
        shutil.rmtree(persist_directory)

    print("Creating new Chroma vector store...")

    batch_size = 50
    vectorstore = None

    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]

        start = i + 1
        end = min(i + batch_size, total_chunks)
        batch_number = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size

        print(
            f"Embedding batch {batch_number}/{total_batches} "
            f"({start}-{end} of {total_chunks})..."
        )

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name="iitj_v1",
                persist_directory=persist_directory,
            )
        else:
            vectorstore.add_documents(batch)

    return vectorstore