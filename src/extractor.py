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

def extract_info(chunks):
    model = OllamaLLM(model="llama3", temperature=0)
    extraction_chain = prompt | model
    results = []
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"正在处理第 {idx}/{len(chunks)} 个分块的信息提取...")
        # 使用链处理文本块
        response = extraction_chain.invoke({"text": chunk})
        
        # 尝试解析JSON响应
        try:
            # 这里我们假设模型会返回有效的JSON格式
            # 如果需要更健壮的解析，可以添加错误处理
            import json
            parsed_result = json.loads(response)
            results.append(parsed_result)
        except Exception as e:
            print(f"解析错误: {e}")
            # 如果解析失败，将原始响应作为文本保存
            results.append({"raw_response": response})
    
    return results
