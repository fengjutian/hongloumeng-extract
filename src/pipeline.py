import json
import time
from summarizer import summarize_all_optimized  # 使用优化版
from loader import load_pdf
from chunker import chunk_text_optimized  # 使用优化版
from extractor import extract_info_batch  # 使用优化版
from utils import load_config
from visualizer import build_character_graph_optimized  # 使用优化版

def main():
    # 记录开始时间
    start_time = time.time()
    
    # 加载配置
    config = load_config()
    model_name = config.get("model", "llama3.1:8b")
    print(f"使用模型: {model_name}")
    
    # 1. 读取PDF
    print("1. 开始读取PDF...")
    text = load_pdf("data/hongloumeng.pdf")
    print(f"PDF读取完成，文本长度: {len(text)} 字符")
    
    # 2. 分块
    print("2. 开始分块处理...")
    chunks = chunk_text_optimized(text, config)
    print(f"分块完成，共 {len(chunks)} 个文本块")
    with open("outputs/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # 3. 信息抽取
    print("3. 开始信息抽取...")
    results = extract_info_batch(chunks, model_name=model_name)
    with open("outputs/entities.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 4. 摘要生成
    print("4. 开始摘要生成...")
    summaries, final_summary = summarize_all_optimized(chunks, model_name=model_name)
    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== 分章节摘要 ===\n")
        f.write("\n".join(summaries))
        f.write("\n\n=== 全书摘要 ===\n")
        f.write(final_summary)
    
    print("✅ 红楼梦 信息抽取 & 摘要生成完成！")
    
    # 5. 可视化人物关系图
    print("5. 开始生成人物关系图...")
    build_character_graph_optimized()

        # 5. 人物关系可视化
    # build_character_graph(entities_path="outputs/entities.json",
    #                       output_html="outputs/character_graph.html")

    # 6. 向量化 & 构建 FAISS
    vectorstore = build_vectorstore(chapters_json="outputs/chapters.json",
                                    index_path="outputs/faiss_index")

    # 7. 问答示例
    print("\n=== 问答示例 ===")
    vectorstore = load_vectorstore(index_path="outputs/faiss_index")
    questions = [
        "贾宝玉和林黛玉的关系是什么？",
        "王熙凤的性格特点是什么？",
        "《红楼梦》中哪些人物属于贾家？"
    ]
    for q in questions:
        ans = ask_question(q, vectorstore)
        print(f"Q: {q}\nA: {ans}\n")
    
    # 记录结束时间
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
