import json
import os
import requests
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import hashlib
import time
import uuid
from typing import List, Dict, Any

from config import QWEN_API_KEY


class MilvusFigureStore:
    def __init__(self, host='localhost', port='19530', qwen_api_key=None, qwen_base_url=None):
        # 连接 Milvus
        connections.connect(host=host, port=port)

        # Qwen API 配置
        self.qwen_api_key = qwen_api_key or os.getenv('QWEN_API_KEY')
        self.qwen_base_url = qwen_base_url or os.getenv('QWEN_BASE_URL',
                                                        'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings')

        # 定义集合 schema - 专门为图信息设计
        self.fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="figure_id", dtype=DataType.VARCHAR, max_length=50),  # 图序号
            FieldSchema(name="figure_name", dtype=DataType.VARCHAR, max_length=500),  # 图名
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=200),  # 所属章节
            FieldSchema(name="conclusion", dtype=DataType.VARCHAR, max_length=2000),  # 结论
            FieldSchema(name="subfigure_count", dtype=DataType.INT64),  # 小图总数
            FieldSchema(name="subfigure_data", dtype=DataType.VARCHAR, max_length=10000),  # 小图数据
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=15000),  # 搜索内容
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]

        self.schema = CollectionSchema(self.fields, "Academic paper collection")
        self.collection_name = "paper_figures"

        # 创建集合
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        self.collection = Collection(self.collection_name, self.schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index("vector", index_params)
        print(f"图信息集合 {self.collection_name} 创建成功")

    def load_figure_data(self, json_file_path: str) -> List[Dict]:
        """从JSON文件加载图数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return []

    def parse_figure_info(self, json_data: Dict) -> List[Dict]:
        """解析图信息JSON数据"""
        figures = []

        for file_name, file_data in json_data.items():
            for item in file_data:
                text_data = item.get("text", {})
                chart_analysis = text_data.get("图表分析", [])

                for chart in chart_analysis:
                    # 构建搜索内容
                    search_content = self.build_search_content(chart, file_name)

                    # 处理小图数据
                    subfigure_data = chart.get("小图数据", {})
                    if isinstance(subfigure_data, list):
                        # 如果是列表格式，转换为字符串
                        subfigure_str = json.dumps(subfigure_data, ensure_ascii=False)
                    else:
                        # 如果是字典格式，直接使用
                        subfigure_str = json.dumps(subfigure_data, ensure_ascii=False) if subfigure_data else "{}"

                    figure_info = {
                        "file_name": file_name,
                        "figure_id": chart.get("图序号", ""),
                        "figure_name": chart.get("图名", ""),
                        "section": chart.get("所属章节", ""),
                        "conclusion": chart.get("结论", ""),
                        "subfigure_count": chart.get("小图总数", 0),
                        "subfigure_data": subfigure_str,
                        "content": search_content,
                        "raw_data": chart  # 保留原始数据用于调试
                    }
                    figures.append(figure_info)

        return figures

    def build_search_content(self, chart_data: Dict, file_name: str) -> str:
        """构建用于搜索的内容"""
        parts = [
            f"文件: {file_name}",
            f"图序号: {chart_data.get('图序号', '')}",
            f"图名: {chart_data.get('图名', '')}",
            f"所属章节: {chart_data.get('所属章节', '')}",
            f"结论: {chart_data.get('结论', '')}",
        ]

        # 添加小图信息
        subfigure_data = chart_data.get("小图数据", {})
        if isinstance(subfigure_data, dict):
            for subfig_key, subfig_desc in subfigure_data.items():
                if subfig_desc:
                    parts.append(f"{subfig_key}: {subfig_desc}")
        elif isinstance(subfigure_data, list):
            for subfig_item in subfigure_data:
                if isinstance(subfig_item, dict):
                    for key, value in subfig_item.items():
                        if value:
                            parts.append(f"{key}: {value}")

        return " | ".join(parts)

    def get_embedding_with_qwen(self, text):
        """使用 Qwen API 获取文本向量"""
        headers = {
            'Authorization': f'Bearer {self.qwen_api_key}',
            'Content-Type': 'application/json'
        }

        # 如果文本太长，进行截断
        if len(text) > 2000:
            text = text[:2000]

        data = {
            "model": "text-embedding-v1",
            "input": text
        }

        try:
            response = requests.post(self.qwen_base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            elif 'output_zip' in result and 'embeddings' in result['output_zip']:
                return result['output_zip']['embeddings'][0]['embedding']
            else:
                print(f"API 响应格式异常: {result}")
                import random
                return [random.uniform(-1, 1) for _ in range(1536)]

        except requests.exceptions.HTTPError as e:
            print(f"HTTP 错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            import random
            return [random.uniform(-1, 1) for _ in range(1536)]
        except Exception as e:
            print(f"获取向量失败: {e}")
            import random
            return [random.uniform(-1, 1) for _ in range(1536)]

    def store_figures(self, json_file_path: str):
        """存储图信息到Milvus"""
        # 加载数据
        json_data = self.load_figure_data(json_file_path)
        if not json_data:
            print("无法加载JSON数据")
            return

        # 解析数据
        figures = self.parse_figure_info(json_data)
        print(f"解析到 {len(figures)} 个图信息")

        if not figures:
            print("没有找到可存储的图信息")
            return

        # 准备批量插入数据
        ids = []
        file_names = []
        figure_ids = []
        figure_names = []
        sections = []
        conclusions = []
        subfigure_counts = []
        subfigure_datas = []
        contents = []
        vectors = []

        success_count = 0

        for i, figure in enumerate(figures):
            try:
                # 生成唯一ID
                entity_id = str(uuid.uuid4())

                # 获取向量
                print(f"正在为图 {i + 1}/{len(figures)} 生成向量: {figure['figure_id']}")
                vector = self.get_embedding_with_qwen(figure['content'])

                # 收集数据
                ids.append(entity_id)
                file_names.append(figure['file_name'])
                figure_ids.append(figure['figure_id'])
                figure_names.append(figure['figure_name'])
                sections.append(figure['section'])
                conclusions.append(figure['conclusion'])
                subfigure_counts.append(figure['subfigure_count'])
                subfigure_datas.append(figure['subfigure_data'])
                contents.append(figure['content'])
                vectors.append(vector)

                success_count += 1

                # 每10条或最后一批插入一次
                if len(ids) >= 10 or i == len(figures) - 1:
                    if ids:  # 确保列表不为空
                        data = [
                            ids, file_names, figure_ids, figure_names, sections,
                            conclusions, subfigure_counts, subfigure_datas, contents, vectors
                        ]

                        try:
                            self.collection.insert(data)
                            print(f"成功插入 {len(ids)} 条图记录")

                            # 清空临时列表
                            ids.clear()
                            file_names.clear()
                            figure_ids.clear()
                            figure_names.clear()
                            sections.clear()
                            conclusions.clear()
                            subfigure_counts.clear()
                            subfigure_datas.clear()
                            contents.clear()
                            vectors.clear()

                        except Exception as insert_error:
                            print(f"插入数据失败: {insert_error}")

                time.sleep(0.5)  # 避免API限流

            except Exception as e:
                print(f"处理图 {figure['figure_id']} 时出错: {e}")
                continue

        # 将数据持久化
        self.collection.flush()

        print(f"\n图信息存储完成!")
        print(f"成功: {success_count}/{len(figures)} 个图信息")
        print(f"失败: {len(figures) - success_count} 个图信息")

    def search_figures(self, query: str, top_k: int = 5):
        """搜索相似的图信息"""
        try:
            # 确保集合已加载
            self.collection.load()

            # 获取查询向量
            query_vector = self.get_embedding_with_qwen(query)

            # 搜索参数
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # 执行搜索
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["file_name", "figure_id", "figure_name", "section",
                               "conclusion", "subfigure_count", "subfigure_data", "content"]
            )

            print(f"\n搜索查询: '{query}'")
            print(f"找到 {len(results[0])} 个相关图信息:\n")

            for i, hit in enumerate(results[0]):
                print(f"{i + 1}. [{hit.entity.get('file_name')}] {hit.entity.get('figure_id')}")
                print(f"   图名: {hit.entity.get('figure_name')}")
                print(f"   章节: {hit.entity.get('section')}")
                print(f"   结论: {hit.entity.get('conclusion')}")
                print(f"   小图数量: {hit.entity.get('subfigure_count')}")
                print(f"   距离: {hit.distance:.4f}")

                # 显示部分小图信息
                subfigure_data = hit.entity.get('subfigure_data', '{}')
                try:
                    subfig_dict = json.loads(subfigure_data)
                    if subfig_dict and isinstance(subfig_dict, dict):
                        print("   小图信息:")
                        for key, value in list(subfig_dict.items())[:2]:  # 只显示前2个小图
                            if value:
                                print(f"     - {key}: {value[:100]}...")
                except:
                    pass

                print()

            return results[0]

        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def search_by_filename(self, filename: str):
        """按文件名搜索图信息"""
        try:
            self.collection.load()

            # 使用query进行精确匹配搜索
            results = self.collection.query(
                expr=f'file_name == "{filename}"',
                output_fields=["file_name", "figure_id", "figure_name", "section",
                               "conclusion", "subfigure_count", "subfigure_data"]
            )

            print(f"\n文件 '{filename}' 中的图信息:")
            print(f"找到 {len(results)} 个图:\n")

            for i, result in enumerate(results):
                print(f"{i + 1}. {result['figure_id']} - {result['figure_name']}")
                print(f"   章节: {result['section']}")
                print(f"   结论: {result['conclusion']}")
                print(f"   小图数量: {result['subfigure_count']}")
                print()

            return results

        except Exception as e:
            print(f"按文件名搜索失败: {e}")
            return []

def insert_milvus_figure():
    # 初始化图信息存储类
    figure_store = MilvusFigureStore(
        host='localhost',
        port='19530',
        qwen_api_key=QWEN_API_KEY
    )

    # 1. 存储图信息
    print("第一步: 存储图信息...")
    figure_store.store_figures('ouput_figure_parser/figure_info.json')  # 替换为您的JSON文件路径

    # 2. 搜索测试
    print("\n第二步: 搜索测试...")

    # 测试搜索1: 按技术术语搜索
    print("测试1: 技术术语搜索")
    figure_store.search_figures("Ta和Zr的EAM势比较", top_k=3)

    # # 测试搜索2: 按材料属性搜索
    # print("测试2: 材料属性搜索")
    # figure_store.search_figures("BCC结构 熔点 熔化焓", top_k=3)
    #
    # # 测试搜索3: 按文件名搜索
    # print("测试3: 按文件名搜索")
    # figure_store.search_by_filename("41467_2025_63221_MOESM1_ESM.pdf")
    #
    # # 测试搜索4: 知识点查询
    # print("测试4: 知识点查询")
    # figure_store.search_figures("找出与QCMP相关的图", top_k=3)
    #
    # # 测试搜索5: 复杂查询
    # print("测试5: 复杂查询")
    # figure_store.search_figures("准晶相形成 玻璃形成能力 结晶动力学", top_k=5)


# 使用示例
if __name__ == "__main__":
    insert_milvus_figure()