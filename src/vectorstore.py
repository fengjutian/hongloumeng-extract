from langchain_community.vectorstores import FAISS
# 更新导入语句，使用新的 langchain_huggingface 包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from utils import get_ollama_base_url, load_config

# 检查faiss是否正确安装
faiss_available = False
try:
    import faiss
    faiss_available = True
    print(f"✅ FAISS库已成功导入，版本: {faiss.__version__}")
except ImportError:
    print("⚠️ FAISS库未安装，尝试导入langchain_community中的FAISS...")

# 添加一个函数用于下载模型，支持重试和代理
def download_model_with_retry(model_name, cache_dir=None, max_retries=3, timeout=300, proxies=None):
    """下载模型并支持重试机制"""
    import huggingface_hub
    
    # 设置代理
    if proxies:
        os.environ['HTTP_PROXY'] = proxies.get('http', '')
        os.environ['HTTPS_PROXY'] = proxies.get('https', '')
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"尝试下载模型 {model_name} (第{retry_count+1}/{max_retries}次)...")
            # 下载配置文件
            config_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                cache_dir=cache_dir,
                force_download=False,
                resume_download=True,
                timeout=timeout
            )
            # 下载模型文件
            model_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                cache_dir=cache_dir,
                force_download=False,
                resume_download=True,
                timeout=timeout
            )
            print(f"模型 {model_name} 下载成功!")
            return True
        except Exception as e:
            retry_count += 1
            print(f"下载失败: {e}")
            if retry_count < max_retries:
                print(f"{5 * retry_count}秒后重试...")
                time.sleep(5 * retry_count)
            else:
                print(f"已达到最大重试次数({max_retries})，下载失败。")
                return False


def build_vectorstore(chapters_json=None, index_path="outputs/faiss_index", 
                     batch_size=100, use_multithreading=True, 
                     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     model_cache_dir=None, use_local_model=False, local_model_path=None):
    """
    将章节文本存入矢量数据库（优化版，支持离线模型）
    
    Args:
        chapters_json: 章节JSON文件路径，如果为None则尝试从chunks.json加载
        index_path: 向量数据库保存路径
        batch_size: 批处理大小，用于控制内存使用
        use_multithreading: 是否使用多线程加速
        embedding_model: 使用的嵌入模型
        model_cache_dir: 模型缓存目录
        use_local_model: 是否使用本地模型
        local_model_path: 本地模型路径
    """
    start_time = time.time()
    
    # 1. 加载章节数据（尝试从不同来源）
    print("正在加载章节数据...")
    
    # 尝试从指定路径加载，如果未指定或文件不存在，则尝试从chunks.json加载
    if chapters_json and os.path.exists(chapters_json):
        with open(chapters_json, "r", encoding="utf-8") as f:
            chapters = json.load(f)
        
        texts = []
        metadatas = []
        
        for chap in chapters:
            texts.append(chap["text"])
            metadatas.append({"title": chap["title"]})
    else:
        # 尝试从chunks.json加载
        chunks_path = "outputs/chunks.json"
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            texts = chunks
            # 为每个chunk创建metadata
            metadatas = [{"title": f"Chunk {i+1}"} for i in range(len(chunks))]
        else:
            raise FileNotFoundError("找不到章节数据文件！请先运行pipeline.py生成chunks.json")
    
    # 2. 初始化嵌入模型（支持本地模型和下载重试）
    print(f"正在初始化嵌入模型...")
    
    # 设置缓存目录
    model_kwargs = {'device': 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    # 配置参数字典
    config_kwargs = {
        'model_kwargs': model_kwargs,
        'encode_kwargs': encode_kwargs
    }
    
    if use_local_model and local_model_path and os.path.exists(local_model_path):
        # 使用本地模型
        print(f"使用本地模型: {local_model_path}")
        config_kwargs['model_name'] = local_model_path
    else:
        # 使用在线模型
        print(f"使用在线模型: {embedding_model}")
        config_kwargs['model_name'] = embedding_model
        
        # 如果指定了缓存目录，添加到参数中
        if model_cache_dir:
            config_kwargs['cache_folder'] = model_cache_dir
            os.makedirs(model_cache_dir, exist_ok=True)
        
        # 尝试预下载模型（如果可能）
        try:
            # 检查是否已经下载了模型
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 尝试创建模型实例，如果模型不存在会自动下载
            print("检查模型是否已下载...")
            # 这一行会触发模型下载（如果不存在）
            _ = SentenceTransformer(embedding_model, cache_folder=model_cache_dir)
        except Exception as e:
            print(f"预下载模型失败，将在初始化HuggingFaceEmbeddings时尝试: {e}")
    
    try:
        embeddings = HuggingFaceEmbeddings(**config_kwargs)
    except Exception as e:
        print(f"初始化嵌入模型失败: {e}")
        print("\n建议解决方案：")
        print("1. 检查网络连接是否稳定")
        print("2. 手动下载模型并使用本地模型路径")
        print("3. 设置代理后重试")
        raise
    
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


def load_vectorstore(index_path="outputs/faiss_index", 
                     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     model_cache_dir=None, use_local_model=False, local_model_path=None):
    """
    加载已保存的FAISS向量数据库（优化版，支持离线模型）
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"向量数据库索引路径不存在: {index_path}")
        
    # 加载嵌入模型
    print(f"正在初始化嵌入模型...")
    
    # 设置缓存目录
    model_kwargs = {'device': 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    # 配置参数字典
    config_kwargs = {
        'model_kwargs': model_kwargs,
        'encode_kwargs': encode_kwargs
    }
    
    if use_local_model and local_model_path and os.path.exists(local_model_path):
        # 使用本地模型
        print(f"使用本地模型: {local_model_path}")
        config_kwargs['model_name'] = local_model_path
    else:
        # 使用在线模型
        print(f"使用在线模型: {embedding_model}")
        config_kwargs['model_name'] = embedding_model
        
        # 如果指定了缓存目录，添加到参数中
        if model_cache_dir:
            config_kwargs['cache_folder'] = model_cache_dir
    
    embeddings = HuggingFaceEmbeddings(**config_kwargs)
    
    # 加载向量数据库
    start_time = time.time()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    end_time = time.time()
    print(f"✅ FAISS 向量数据库已从 {index_path} 加载 (耗时: {end_time - start_time:.2f} 秒)")
    return vectorstore


def create_qa_chain(vectorstore, model_name="llama3:latest", config=None):
    """
    创建基于向量数据库的问答链
    """
    # 加载配置（如果未提供）
    if config is None:
        config = load_config()
    
    # 初始化Ollama LLM
    llm = OllamaLLM(
        model=model_name,
        base_url=get_ollama_base_url(config),
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
                        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                        model_cache_dir=None, use_local_model=False, local_model_path=None):
    """
    使用向量数据库进行问答（支持离线模型）
    """
    # 加载配置
    config = load_config()
    
    # 加载向量数据库
    vectorstore = load_vectorstore(
        index_path, 
        embedding_model, 
        model_cache_dir, 
        use_local_model, 
        local_model_path
    )
    
    # 创建问答链（传递config参数）
    qa_chain = create_qa_chain(vectorstore, model_name, config)
    
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
    # 配置参数
    use_local_model = False  # 设置为True使用本地模型
    local_model_path = "path/to/local/model"  # 本地模型路径
    model_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models")  # 模型缓存目录
    
    try:
        # 检查faiss是否安装
        import faiss
        print(f"✅ FAISS库已成功安装，版本: {faiss.__version__}")
    except ImportError:
        print("❌ FAISS库未安装！请先运行以下命令安装：")
        print("pip install faiss-cpu")
        import sys
        sys.exit(1)
    
    try:
        # 如果向量数据库已存在，可以直接进行问答
        if os.path.exists("outputs/faiss_index"):
            question = "红楼梦的主要人物有哪些？"
            result = qa_with_vectorstore(
                question, 
                model_cache_dir=model_cache_dir,
                use_local_model=use_local_model,
                local_model_path=local_model_path
            )
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
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 可以换更轻量的模型
                model_cache_dir=model_cache_dir,
                use_local_model=use_local_model,
                local_model_path=local_model_path
            )
            print("向量数据库构建完成！")
    except Exception as e:
        print(f"执行问答时出错: {e}")
        # 提供更友好的错误提示
        if "找不到章节数据文件" in str(e):
            print("\n请按照以下步骤操作：")
            print("1. 确保Ollama服务已启动（运行 'ollama serve'）")
            print("2. 确保已下载llama3:latest模型（运行 'ollama pull llama3:latest'）")
            print("3. 先运行pipeline.py生成必要的数据文件：")
            print("   python src/pipeline.py")
            print("4. 然后再运行vectorstore.py")
        elif "ConnectionResetError" in str(e) or "Read timed out" in str(e):
            print("\n模型下载失败，可能是网络连接问题。建议尝试以下解决方案：")
            print("1. 检查网络连接是否稳定，尝试使用VPN或代理")
            print("2. 手动下载sentence-transformers模型，然后使用本地模型路径：")
            print("   - 将use_local_model设置为True")
            print("   - 设置local_model_path为本地模型的路径")
            print("3. 尝试使用其他轻量级的嵌入模型")
        elif "Could not import faiss" in str(e):
            print("\n缺少FAISS库，请运行以下命令安装：")
            print("pip install faiss-cpu")
        elif "Could not import sentence_transformers" in str(e):
            print("\n缺少必要的依赖包，请运行以下命令安装：")
            print("pip install sentence-transformers langchain-huggingface")
