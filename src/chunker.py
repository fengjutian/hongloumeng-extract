from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    return splitter.split_text(text)
