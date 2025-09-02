from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import concurrent.futures
from utils import get_ollama_base_url

def summarize_chunk(chunk: str, model_name="llama3:latest", config=None) -> str:
    """对单个文本块生成摘要"""
    base_url = get_ollama_base_url(config) if config else "http://localhost:11434"
    model = OllamaLLM(model=model_name, temperature=0, base_url=base_url)
    prompt = PromptTemplate.from_template(
        "请为以下文本生成简洁的中文摘要（重点人物、情节走向）：\n\n{chunk}\n\n摘要："
    )
    chain = prompt | model
    return chain.invoke({"chunk": chunk})

def summarize_all_optimized(chunks, model_name="llama3.1:8b", batch_size=4, config=None):
    """批量摘要，并汇总为整体摘要（优化版）"""
    base_url = get_ollama_base_url(config) if config else "http://localhost:11434"
    # 初始化模型一次
    model = OllamaLLM(model=model_name, temperature=0, base_url=base_url)
    prompt = PromptTemplate.from_template(
        "请为以下文本生成简洁的中文摘要（重点人物、情节走向）：\n\n{chunk}\n\n摘要："
    )
    chain = prompt | model
    summaries = []
    
    # 使用线程池并行处理，但限制并发数量
    actual_batch_size = min(batch_size, len(chunks), 2)  # 进一步限制并发数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_batch_size) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(chain.invoke, {"chunk": chunk}): idx 
            for idx, chunk in enumerate(chunks)
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                summary = future.result()
                summaries.append((idx, summary))
                print(f"已完成第 {idx+1}/{len(chunks)} 个分块的摘要...")
            except Exception as e:
                print(f"处理第 {idx+1} 个分块时出错: {e}")
                summaries.append((idx, "[摘要生成失败]"))
    
    # 按原始顺序排序
    summaries.sort(key=lambda x: x[0])
    sorted_summaries = [summary for idx, summary in summaries]
    
    # 将所有小摘要拼接成大摘要
    combined_text = "\n".join(sorted_summaries)
    
    # 用模型再生成一个整体摘要
    final_prompt = f"以下是《红楼梦》分章节摘要，请综合整理成一份完整的全书摘要：\n\n{combined_text}\n\n完整摘要："
    final_summary = model.invoke(final_prompt)
    
    return sorted_summaries, final_summary
