
from loader import split_pdf

# 拆分PDF（每20页一个文件）
split_files = split_pdf(
    "d:\\GitHub\\hongloumeng-extract\\data\\hongloumeng.pdf",
    output_dir="d:\\GitHub\\hongloumeng-extract\\data\\split_pdfs",
    pages_per_file=20
)

# 查看生成的文件
print(split_files)