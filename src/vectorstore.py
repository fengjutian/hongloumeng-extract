from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os

def build_vectorstore(chapters_json="outputs/chapters.json", index_path="outputs/faiss_index"):
    """
    将章节文本存入矢量数据库
    """
    # 1. 加载章节
    with open(chapters_json, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    texts = []
    metadatas = []

    for chap in chapters:
        texts.append(chap["text"])
        metadatas.append({"title": chap["title"]})

    # 2. 生成向量
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. 构建 FAISS 向量数据库
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    # 4. 保存索引
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"✅ FAISS 向量数据库已保存到 {index_path}")
    return vectorstore
