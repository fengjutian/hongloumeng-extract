from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from utils import get_ollama_base_url


def build_vectorstore(chapters_json="outputs/chapters.json", index_path="outputs/faiss_index", 
                     batch_size=100, use_multithreading=True, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    将章节文本存入矢量数据库（优化版）
    
    Args:
        chapters_json: 章节JSON文件路径
        index_path: 向量数据库保存路径
        batch_size: 批处理大小，用于控制内存使用
        use_multithreading: 是否使用多线程加速
        embedding_model: 使用的嵌入模型
    """
    start_time = time.time()
    
    # 1. 加载章节（使用更快的JSON解析）
    print("正在加载章节数据...")
    with open(chapters_json, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    
    texts = []
    metadatas = []
    
    for chap in chapters:
        texts.append(chap["text"])
        metadatas.append({"title": chap["title"]})
    
    # 2. 初始化嵌入模型（添加缓存参数）
    print(f"正在初始化嵌入模型: {embedding_model}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 3. 构建 FAISS 向量数据库（优化版本）
    print("正在构建向量数据库...")
    
    # 小数据集直接处理
    if len(texts) <= batch_size:
        vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    else:
        # 大数据集使用批处理
        vectorstore = None
        
        # 如果启用多线程，使用并行批处理
        if use_multithreading:
            # 创建批处理函数
            def process_batch(batch_texts, batch_metadatas):
                return FAISS.from_texts(batch_texts, embedding=embeddings, metadatas=batch_metadatas)
            
            # 准备批处理数据
            batches = []
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batches.append((texts[i:end_idx], metadatas[i:end_idx]))
            
            # 并行处理批数据
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                batch_results = list(executor.map(lambda x: process_batch(x[0], x[1]), batches))
            
            # 合并结果
            vectorstore = batch_results[0]
            for i in range(1, len(batch_results)):
                vectorstore.merge_from(batch_results[i])
        else:
            # 不使用多线程，使用单线程批处理
            vectorstore = FAISS.from_texts(texts[:batch_size], embedding=embeddings, metadatas=metadatas[:batch_size])
            
            # 分批次添加剩余文本
            for i in range(batch_size, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                vectorstore.add_texts(texts[i:end_idx], metadatas[i:end_idx])
    
    # 4. 保存索引（使用更快的文件操作）
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    
    end_time = time.time()
    print(f"✅ FAISS 向量数据库已保存到 {index_path}")
    print(f"⏱️  构建时间: {end_time - start_time:.2f} 秒")
    
    return vectorstore


def load_vectorstore(index_path="outputs/faiss_index", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    加载已保存的FAISS向量数据库（优化版）
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"向量数据库索引路径不存在: {index_path}")
        
    # 加载嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 加载向量数据库
    start_time = time.time()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    end_time = time.time()
    print(f"✅ FAISS 向量数据库已从 {index_path} 加载 (耗时: {end_time - start_time:.2f} 秒)")
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
    
    # 创建检索器（优化检索参数）
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # 使用MMR替代similarity，提高结果多样性
        search_kwargs={"k": 3, "fetch_k": 20}  # 先获取更多候选再筛选
    )
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


def qa_with_vectorstore(question, index_path="outputs/faiss_index", model_name="llama3:latest", 
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    使用向量数据库进行问答
    """
    # 加载向量数据库
    vectorstore = load_vectorstore(index_path, embedding_model)
    
    # 创建问答链
    qa_chain = create_qa_chain(vectorstore, model_name)
    
    # 执行问答
    start_time = time.time()
    result = qa_chain.invoke({"query": question})
    end_time = time.time()
    
    # 格式化结果
    answer = {
        "question": question,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
        "response_time": end_time - start_time
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
            print(f"响应时间: {result['response_time']:.2f} 秒")
            print("\n来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['title']}")
        else:
            print("向量数据库不存在，正在构建...")
            # 可以通过调整参数进一步优化性能
            vectorstore = build_vectorstore(
                batch_size=200,  # 根据系统内存调整
                use_multithreading=True,  # 启用多线程
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 可以换更轻量的模型
            )
            print("向量数据库构建完成！")
    except Exception as e:
        print(f"执行问答时出错: {e}")
