from loader import split_pdf
import os

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

if __name__ == "__main__":
    # 设置输入PDF路径
    pdf_path = "d:\\GitHub\\hongloumeng-extract\\data\\hongloumeng.pdf"
    
    # 设置输出目录
    output_dir = "d:\\GitHub\\hongloumeng-extract\\data\\split_pdfs"
    
    # 设置每个文件包含的页数
    pages_per_file = 20  # 可以根据需要调整这个值
    
    # 提示用户
    print(f"开始拆分PDF文件: {pdf_path}")
    print(f"每个文件包含 {pages_per_file} 页")
    print(f"输出目录: {output_dir}")
    
    try:
        # 调用拆分函数
        split_files = split_pdf(
            pdf_path,
            output_dir=output_dir,
            pages_per_file=pages_per_file
        )
        
        # 打印结果
        print(f"PDF拆分完成! 共生成 {len(split_files)} 个文件:")
        for file_path in split_files:
            print(f"- {file_path}")
            
    except Exception as e:
        print(f"拆分过程中出错: {str(e)}")