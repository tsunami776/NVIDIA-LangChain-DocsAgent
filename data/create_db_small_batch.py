import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "docs-agent/app/chroma_small"
DATA_PATH = "md_files/langchain_docs_v0.1_small"
BATCH_SIZE = 1000  # Adjust batch size as needed

def main():
    documents = load_documents()
    chunks = split_text(documents)
    create_initial_db(chunks[:BATCH_SIZE])
    add_batches_to_db(chunks[BATCH_SIZE:])

def load_documents():
    markdown_files = collect_markdown_files(DATA_PATH)
    documents = []
    for file_path in markdown_files:
        loader = DirectoryLoader(os.path.dirname(file_path), glob=os.path.basename(file_path))
        documents.extend(loader.load())
    return documents

def collect_markdown_files(directory):
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

    return chunks

def create_initial_db(initial_batch: list[Document]):
    # Clear out the database directory first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Filter out empty chunks
    non_empty_chunks = [chunk for chunk in initial_batch if chunk.page_content.strip()]

    if not non_empty_chunks:
        print("No non-empty chunks to process.")
        return

    embeddings = NVIDIAEmbeddings()

    try:
        print("Creating initial database")

        # Create the initial Chroma database
        db = Chroma.from_documents(
            documents=non_empty_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        db.persist()
        print(f"Saved initial batch of {len(non_empty_chunks)} documents to {CHROMA_PATH}.")
    except Exception as e:
        print(f"Error creating initial database: {e}")

def add_batches_to_db(chunks: list[Document]):
    embeddings = NVIDIAEmbeddings()

    # Filter out empty chunks
    non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not non_empty_chunks:
        print("No non-empty chunks to process.")
        return

    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    total_batches = (len(non_empty_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    # Process remaining batches
    for i in range(0, len(non_empty_chunks), BATCH_SIZE):
        batch = non_empty_chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk.page_content for chunk in batch]

        if not batch_texts:
            print(f"Skipping empty batch at index {i}")
            continue

        try:
            current_batch_number = (i // BATCH_SIZE) + 1
            print(f"Processing batch {current_batch_number}/{total_batches}")
            db.add_documents(batch)
            print(f"Added batch {current_batch_number}/{total_batches} to the database")
        except Exception as e:
            print(f"Error adding batch at index {i}: {e}")
            continue

if __name__ == "__main__":
    main()
