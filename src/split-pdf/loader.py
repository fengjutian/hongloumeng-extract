from pypdf import PdfReader, PdfWriter
import os

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def split_pdf(input_path: str, output_dir: str = None, pages_per_file: int = 10, start_page: int = 0, end_page: int = None):
    """
    将PDF文件分拆成多个小PDF文件
    
    参数:
        input_path: 输入PDF文件路径
        output_dir: 输出目录，默认为输入文件所在目录
        pages_per_file: 每个输出文件包含的页数
        start_page: 起始页码（从0开始）
        end_page: 结束页码（从0开始），None表示到文档末尾
    
    返回:
        生成的文件路径列表
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入PDF
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    
    # 确定页码范围
    if end_page is None or end_page >= total_pages:
        end_page = total_pages - 1
    
    # 验证页码范围
    if start_page < 0 or end_page < start_page:
        raise ValueError("无效的页码范围")
    
    # 生成输出文件名基础
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 分拆PDF
    output_files = []
    current_page = start_page
    file_index = 1
    
    while current_page <= end_page:
        # 计算当前文件的结束页码
        file_end_page = min(current_page + pages_per_file - 1, end_page)
        
        # 创建PDF写入器
        writer = PdfWriter()
        
        # 添加页面到当前文件
        for page_num in range(current_page, file_end_page + 1):
            writer.add_page(reader.pages[page_num])
        
        # 生成输出文件名
        output_path = os.path.join(output_dir, f"{name_without_ext}_part{file_index}_{current_page+1}-{file_end_page+1}.pdf")
        
        # 保存当前文件
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        # 添加到输出文件列表
        output_files.append(output_path)
        
        # 更新计数器
        current_page = file_end_page + 1
        file_index += 1
    
    return output_files
