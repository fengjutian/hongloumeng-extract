from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List

# 定义抽取的数据结构
class Character(BaseModel):
    name: str
    role: str

class Event(BaseModel):
    summary: str
    characters_involved: List[str]

class Location(BaseModel):
    name: str
    description: str

# 创建提取提示模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个专业的信息提取算法。请从提供的文本中提取相关信息。\n"
        "请以JSON格式输出提取的信息，包含三个部分：characters（人物列表）、events（事件列表）和locations（地点列表）。\n"
        "每个部分应包含符合定义的数据结构。如果无法确定某个属性的值，请返回null。\n"
        "请勿添加任何额外的解释或说明文字。"
    ),
    ("human", "{text}")
])

def extract_info_batch(chunks, model_name="llama3.1:8b", batch_size=5):
    """批量处理文本块以提高性能"""
    try:
        model = OllamaLLM(model=model_name, temperature=0)
        extraction_chain = prompt | model
        results = []
        
        # 分批处理
        for batch_idx in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_idx:batch_idx+batch_size]
            batch_results = []
            
            # 并行处理每个批次内的文本块
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_chunk = {executor.submit(extraction_chain.invoke, {"text": chunk}): chunk for chunk in batch_chunks}
                
                for idx, future in enumerate(concurrent.futures.as_completed(future_to_chunk), 1):
                    chunk_idx = batch_idx + idx
                    print(f"正在处理第 {chunk_idx}/{len(chunks)} 个分块的信息提取...")
                    try:
                        response = future.result()
                        # 解析响应...
                        try:
                            import json
                            parsed_result = json.loads(response)
                            batch_results.append(parsed_result)
                        except Exception as e:
                            print(f"解析错误: {e}")
                            batch_results.append({"raw_response": response})
                    except Exception as e:
                        print(f"处理批处理块时出错: {e}")
            
            results.extend(batch_results)
        
        return results
    except Exception as e:
        print(f"模型调用错误: {e}")
        print("提示：请确保你已经安装了Ollama并下载了相应的模型")
        print(f"当前尝试使用的模型: {model_name}")
        print("你可以使用 'ollama list' 命令查看已安装的模型")
        print("如果需要下载模型，可以使用 'ollama pull 模型名称' 命令")
        raise
