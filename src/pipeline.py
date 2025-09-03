import time
import json  # 添加缺失的json模块导入
from summarizer import summarize_all_optimized  # 使用优化版
from loader import load_pdf
from chunker import chunk_text_optimized  # 使用优化版
from extractor import extract_info_batch  # 使用优化版
from utils import load_config, check_ollama_connection, wait_for_ollama_service, get_ollama_base_url
from visualizer import build_character_graph_optimized  # 使用优化版

def main():
    # 记录开始时间
    start_time = time.time()
    
    # 加载配置
    config = load_config()
    model_name = config.get("model", "llama3:latest")
    base_url = get_ollama_base_url(config)
    batch_size = config.get("batch_size", 2)
    print(f"使用模型: {model_name}")
    print(f"Ollama服务地址: {base_url}")
    
    # 检查Ollama服务连接
    if not check_ollama_connection(base_url):
        print("Ollama服务未启动，正在等待服务启动...")
        if not wait_for_ollama_service(base_url):
            print("错误：无法连接到Ollama服务。请确保Ollama已安装并正在运行。")
            print("你可以在另一个命令行窗口中运行 'ollama serve' 来启动服务。")
            return
    
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
    results = extract_info_batch(chunks, model_name=model_name, batch_size=batch_size, config=config)
    with open("outputs/entities.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 4. 摘要生成
    print("4. 开始摘要生成...")
    summaries, final_summary = summarize_all_optimized(chunks, model_name=model_name, batch_size=batch_size, config=config)
    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== 分章节摘要 ===\n")
        f.write("\n".join(summaries))
        f.write("\n\n=== 全书摘要 ===\n")
        f.write(final_summary)
    
    print("✅ 红楼梦 信息抽取 & 摘要生成完成！")
    
    # 5. 可视化人物关系图
    print("5. 开始生成人物关系图...")
    build_character_graph_optimized()

    # 记录结束时间
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
