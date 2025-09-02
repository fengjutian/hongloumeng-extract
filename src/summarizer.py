from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

def summarize_chunk(chunk: str, model_name="llama3") -> str:
    """对单个文本块生成摘要"""
    model = OllamaLLM(model=model_name, temperature=0)
    prompt = PromptTemplate.from_template(
        "请为以下文本生成简洁的中文摘要（重点人物、情节走向）：\n\n{chunk}\n\n摘要："
    )
    chain = prompt | model
    return chain.invoke({"chunk": chunk})

def summarize_all(chunks, model_name="llama3"):
    """批量摘要，并汇总为整体摘要"""
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"正在处理第 {idx}/{len(chunks)} 个分块...")
        summary = summarize_chunk(chunk, model_name=model_name)
        summaries.append(summary)

    # 将所有小摘要拼接成大摘要
    combined_text = "\n".join(summaries)

    # 用模型再生成一个整体摘要
    model = OllamaLLM(model=model_name, temperature=0)
    final_prompt = f"以下是《红楼梦》分章节摘要，请综合整理成一份完整的全书摘要：\n\n{combined_text}\n\n完整摘要："
    final_summary = model.invoke(final_prompt)

    return summaries, final_summary
