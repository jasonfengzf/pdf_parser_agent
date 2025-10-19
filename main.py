from parser_by_MinerU import parser_by_MinerU
from parser_by_reducto import parser_by_reducto
from parser_figure_by_qwen import parser_figure_by_qwen
from insert_milvus_paper import insert_milvus_paper
from insert_milvus_content import insert_milvus_content
from insert_milvus_figure import insert_milvus_figure
from pdf_multi_agent import pdf_agent

def main():
    print("正在通过MinerU API 解析文件...")
    parser_by_MinerU()
    print("正在通过reducto API 解析文件...")
    parser_by_reducto()
    print("正在通过qwen API 解析文件...")
    parser_figure_by_qwen()
    print("正在插入论文...")
    insert_milvus_paper()
    print("正在插入论文内容...")
    insert_milvus_content()
    print("正在插入论文图片信息...")
    insert_milvus_figure()
    print("正在启动问答系统...")
    pdf_agent()


if __name__ == "__main__":
    main()

