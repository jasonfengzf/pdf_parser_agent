import requests
import base64
import os
from PIL import Image
import io
import json
import glob
import re

from config import QWEN_API_KEY

# 尝试使用不同的PDF处理库
try:
    import fitz  # PyMuPDF

    USE_PYMUPDF = True
except ImportError:
    try:
        from pdf2image import convert_from_path

        USE_PYMUPDF = False
    except ImportError:
        print("请安装 PDF 处理库：")
        print("pip install pymupdf")
        print("或")
        print("pip install pdf2image poppler-utils")
        exit(1)


class QwenVLPdfParser:
    def __init__(self, api_key):
        """
        初始化Qwen-VL PDF解析器
        """
        self.api_key = f"{QWEN_API_KEY}"
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def pdf_to_images(self, pdf_path, dpi=150):
        """
        将PDF转换为图像列表
        """
        images = []
        try:
            if USE_PYMUPDF:
                # 使用 PyMuPDF
                pdf_document = fitz.open(pdf_path)

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)

                pdf_document.close()
            else:
                # 使用 pdf2image
                images = convert_from_path(pdf_path, dpi=dpi)

            print(f"成功转换 {len(images)} 页PDF为图像")

        except Exception as e:
            print(f"PDF转换错误: {e}")
            # 备用方案：如果PDF转换失败，尝试使用其他方法
            try:
                import pdfplumber
                print("尝试使用 pdfplumber 进行文本提取...")
                # 这里可以添加文本提取逻辑
            except ImportError:
                print("建议安装 pdfplumber: pip install pdfplumber")

        return images

    def image_to_base64(self, image):
        """
        将PIL图像转换为base64编码
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def extract_json_from_response(self, response_data):
        """
        从API响应中提取并转换JSON数据
        """
        try:
            # 如果响应是列表，处理其中的字典
            if isinstance(response_data, list):
                processed_list = []
                for item in response_data:
                    if isinstance(item, dict) and "text" in item:
                        # 处理text字段中的JSON字符串
                        text_content = item["text"]
                        if isinstance(text_content, str):
                            # 提取JSON部分
                            json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group()
                                try:
                                    json_data = json.loads(json_str)
                                    processed_list.append({"text": json_data})
                                except json.JSONDecodeError:
                                    # 如果JSON解析失败，保留原始文本
                                    processed_list.append(item)
                            else:
                                # 如果没有找到JSON，保留原始内容
                                processed_list.append(item)
                        else:
                            # 如果text不是字符串，直接保留
                            processed_list.append(item)
                    else:
                        # 如果不是包含text的字典，直接保留
                        processed_list.append(item)
                return processed_list

            # 如果响应是字典，直接返回
            elif isinstance(response_data, dict):
                return response_data

            # 如果是字符串，尝试提取JSON
            elif isinstance(response_data, str):
                json_match = re.search(r'\{.*\}', response_data, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
                return response_data

            # 其他情况直接返回
            else:
                return response_data

        except Exception as e:
            print(f"JSON提取失败: {e}")
            return response_data

    def call_qwen_vl(self, prompt, images):
        """
        调用Qwen-VL模型进行解析
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构建消息内容
        messages = []

        # 添加图像（限制前5页以避免token过多）
        for img in images[:5]:  # 限制处理前5页以避免API限制
            base64_img = self.image_to_base64(img)
            messages.append({
                "image": f"data:image/png;base64,{base64_img}"
            })

        # 添加文本提示
        messages.append({
            "text": prompt
        })

        data = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": messages
                    }
                ]
            },
            "parameters": {
                "max_tokens": 2048
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            # 提取并处理响应内容
            if result and "output" in result:
                content = result["output"]["choices"][0]["message"]["content"]
                # 处理响应内容，提取JSON数据
                processed_content = self.extract_json_from_response(content)
                return processed_content
            else:
                return None

        except Exception as e:
            print(f"API调用错误: {e}")
            return None

    def parse_pdf_content(self, pdf_path, prompt=None):
        """
        解析单个PDF内容
        """
        if prompt is None:
            prompt = """请仔细分析这个PDF文档的内容，包括：
            1.抽取图表，给出图的总数，然后再对每个图进行结构化，获取名称、结论、所属章节等信息
            2.逐一分析每个图里面的小图，给出小图总数，然后用字典形式列出所有小图的数据，没有数据则跳过
            请用清晰的结构化格式json返回分析结果。输出格式：{"图总数"：,"图表分析":[{"图序号":,"图名":"","所属章节":"","结论":"","小图总数":,"小图数据":[{"小图1":"","小图2":""}]}]}
            注意：请尽可能用中文进行表述
            """

        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            return f"文件不存在: {pdf_path}"

        print(f"正在处理: {os.path.basename(pdf_path)}")

        # 转换PDF为图像
        images = self.pdf_to_images(pdf_path)

        if not images:
            return "PDF转换失败，请检查PDF文件是否有效"

        # 调用Qwen-VL模型
        result = self.call_qwen_vl(prompt, images)

        if result:
            return result
        else:
            return "解析失败，请检查API密钥和网络连接"

    def batch_parse_pdfs(self, folder_path, output_dir="output", prompt=None):
        """
        批量处理文件夹中的所有PDF文件

        Args:
            folder_path: 包含PDF文件的文件夹路径
            output_dir: 输出结果保存的目录
            prompt: 解析提示词
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 查找文件夹中的所有PDF文件
        pdf_pattern = os.path.join(folder_path, "*.pdf")
        pdf_files = glob.glob(pdf_pattern)

        if not pdf_files:
            print(f"在文件夹 {folder_path} 中未找到PDF文件")
            return

        print(f"找到 {len(pdf_files)} 个PDF文件，开始批量处理...")

        results = {}

        for pdf_file in pdf_files:
            try:
                # 解析单个PDF
                result = self.parse_pdf_content(pdf_file, prompt)

                # 保存结果
                filename = os.path.basename(pdf_file)
                results[filename] = result

                # 保存到单独的文件
                output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"PDF文件: {filename}\n")
                    f.write("解析结果:\n")

                    # 根据结果类型进行不同的写入处理
                    if isinstance(result, (dict, list)):
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    else:
                        f.write(str(result))

                    f.write("\n" + "=" * 50 + "\n")

                print(f"已完成: {filename} -> {output_file}")

            except Exception as e:
                error_msg = f"处理文件 {pdf_file} 时出错: {str(e)}"
                print(error_msg)
                results[os.path.basename(pdf_file)] = error_msg

        # 保存汇总结果
        summary_file = os.path.join(output_dir, "figure_info.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"汇总结果已保存至: {summary_file}")
        except Exception as e:
            print(f"保存汇总结果失败: {e}")
            # 尝试处理无法序列化的对象
            serializable_results = {}
            for filename, result in results.items():
                try:
                    # 测试是否可序列化
                    json.dumps(result, ensure_ascii=False)
                    serializable_results[filename] = result
                except:
                    # 如果不可序列化，转换为字符串
                    serializable_results[filename] = str(result)

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"处理后的汇总结果已保存至: {summary_file}")

        print(f"\n批量处理完成！")
        print(f"所有单个结果已保存至: {output_dir} 目录")

        return results


# 使用示例
def parser_figure_by_qwen():
    API_KEY = ""

    # 创建解析器实例
    parser = QwenVLPdfParser(API_KEY)

    # 批量处理
    folder_path = "pdf_file"
    output_dir = "ouput_figure_parser"

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    # 批量处理
    parser.batch_parse_pdfs(folder_path, output_dir)


if __name__ == "__main__":
    parser_figure_by_qwen()