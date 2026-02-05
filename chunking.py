from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []
    for doc in documents:
        parts = splitter.split_text(doc["content"])
        for p in parts:
            chunks.append({
                "content": p,
                "source": doc["source"]
            })

    return chunks
