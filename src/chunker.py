from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def chunk_text_optimized(text: str, config=None):
    """智能分块策略"""
    if config is None:
        config = {}
        
    # 从配置中获取参数或使用默认值
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 100)
    
    # 对于非常长的文档，增加chunk_size以减少总块数
    if len(text) > 100000:
        chunk_size = 2000
        chunk_overlap = 200
        
    # 检测文本是否包含明显的章节结构
    if re.search(r'第[一二三四五六七八九十百]+回', text):
        # 如果有明显章节，优先按章节分块
        chapters = re.split(r'(第[一二三四五六七八九十百]+回)', text)
        # 组合章节标题和内容
        combined_chapters = []
        for i in range(1, len(chapters), 2):
            title = chapters[i]
            content = chapters[i+1] if i+1 < len(chapters) else ""
            combined_chapters.append(title + content)
        
        # 对长章节进行二次分块
        result = []
        for chapter in combined_chapters:
            if len(chapter) <= chunk_size * 1.5:
                result.append(chapter)
            else:
                # 对过长的章节使用递归字符分割
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", "。", "，", " "]
                )
                result.extend(splitter.split_text(chapter))
        
        return result
    
    # 没有明显章节结构时，使用原始的递归字符分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    return splitter.split_text(text)
