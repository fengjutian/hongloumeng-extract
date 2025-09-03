from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import json
import os
from utils import get_ollama_base_url


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


def load_vectorstore(index_path="outputs/faiss_index"):
    """
    加载已保存的FAISS向量数据库
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"向量数据库索引路径不存在: {index_path}")
        
    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 加载向量数据库
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ FAISS 向量数据库已从 {index_path} 加载")
    return vectorstore


def create_qa_chain(vectorstore, model_name="llama3:latest"):
    """
    创建基于向量数据库的问答链
    """
    # 初始化Ollama LLM
    llm = OllamaLLM(
        model=model_name,
        base_url=get_ollama_base_url(),
        temperature=0
    )
    
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


def qa_with_vectorstore(question, index_path="outputs/faiss_index", model_name="llama3:latest"):
    """
    使用向量数据库进行问答
    """
    # 加载向量数据库
    vectorstore = load_vectorstore(index_path)
    
    # 创建问答链
    qa_chain = create_qa_chain(vectorstore, model_name)
    
    # 执行问答
    result = qa_chain.invoke({"query": question})
    
    # 格式化结果
    answer = {
        "question": question,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
    
    return answer


if __name__ == "__main__":
    # 示例：使用问答功能
    try:
        # 如果向量数据库已存在，可以直接进行问答
        if os.path.exists("outputs/faiss_index"):
            question = "红楼梦的主要人物有哪些？"
            result = qa_with_vectorstore(question)
            print(f"\n问题: {result['question']}")
            print(f"回答: {result['answer']}")
            print("\n来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['title']}")
        else:
            print("向量数据库不存在，请先运行pipeline.py构建向量数据库")
            # 提示用户如何解决
            print("\n请按照以下步骤操作:")
            print("1. 确保Ollama服务已启动 (运行 'ollama serve')")
            print("2. 确保已下载llama3:latest模型 (运行 'ollama pull llama3:latest')")
            print("3. 运行 'python src/pipeline.py' 构建向量数据库")
    except Exception as e:
        print(f"执行问答时出错: {e}")
