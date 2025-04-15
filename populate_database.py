import os
import shutil
import argparse
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from _1_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def clear_database():
    """Removes the ChromaDB directory to reset the stored embeddings."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def load_documents():
    """Loads PDF documents from the specified directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    """Splits large documents into smaller chunks for efficient embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]):
    """Generates unique IDs for each document chunk."""

    last_page_ID = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Update index if still on same page
        if current_page_id == last_page_ID:
            current_chunk_index += 1
            
        else:
            current_chunk_index = 0

        # Assign unique chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_ID = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document]):
    """Embeds and stores document chunks in ChromaDB avoiding duplicates."""

    # Initialize ChromaDB with embedding function
    db = Chroma(
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH
    )

    # Assign uniwue IDs to chunks, using the above function
    chunk_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing document IDs from ChromaDB
    existing_items = db.get(include=[]) # IDs are always included by default
    existing_ids = set(existing_items['ids'])
    print(f"Number of existing document IDs in DB: {len(existing_ids)}")

    # Filter new chunks that are not already stored.
    new_chunks = [chunk for chunk in chunk_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids = new_chunk_ids)
    else:
        print("✅ No new documents to add")
    print("📦 Chunk IDs and content preview (all input chunks):")
    for chunk in chunk_with_ids:
        print(f"ID: {chunk.metadata['id']} | Content: {chunk.page_content[:60]}...\n")


def main():
    """Main function to control document processing and vector storage."""
    
    # Handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action= "store_true", help="Reset the database.")
    args = parser.parse_args()

    # If reset flag is provided, clear the existing database
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Process documents: Load, split and store in ChromaDB
    documents = load_documents()
    chunks = split_documents(documents=documents)
    add_to_chroma(chunks=chunks)

if __name__=="__main__":
    main()