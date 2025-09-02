import json
from summarizer import summarize_all
from loader import load_pdf
from chunker import chunk_text
from extractor import extract_info

def main():
    # 1. 读取PDF
    text = load_pdf("data/hongloumeng.pdf")

    # 2. 分块
    chunks = chunk_text(text)
    with open("outputs/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 3. 信息抽取
    results = extract_info(chunks)
    with open("outputs/entities.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 4. 摘要生成
    summaries, final_summary = summarize_all(chunks)
    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== 分章节摘要 ===\n")
        f.write("\n".join(summaries))
        f.write("\n\n=== 全书摘要 ===\n")
        f.write(final_summary)

    print("✅ 红楼梦 信息抽取 & 摘要生成完成！")

if __name__ == "__main__":
    main()
