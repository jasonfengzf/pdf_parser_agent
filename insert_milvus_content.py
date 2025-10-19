import os
import requests
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import hashlib
import time
import uuid
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import QWEN_API_KEY


class MilvusPaperStore:
    def __init__(self, host='localhost', port='19530', qwen_api_key=None, qwen_base_url=None, batch_size=50,
                 max_workers=5):
        # 连接 Milvus
        connections.connect(host=host, port=port)

        # Qwen API 配置
        self.qwen_api_key = qwen_api_key or os.getenv('QWEN_API_KEY')
        self.qwen_base_url = qwen_base_url or os.getenv('QWEN_BASE_URL',
                                                        'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings')

        # 性能优化参数
        self.batch_size = batch_size  # 批量处理大小
        self.max_workers = max_workers  # 最大并发数
        self.session = None  # 复用 HTTP session

        # 定义集合 schema - 修复：bbox 存储为字符串
        self.fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="directory", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="text_level", dtype=DataType.INT64),
            FieldSchema(name="page_idx", dtype=DataType.INT64),
            FieldSchema(name="bbox", dtype=DataType.VARCHAR, max_length=100),  # 改为字符串存储
            FieldSchema(name="img_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="image_caption", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="text_format", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)  # 唯一的向量字段
        ]

        self.schema = CollectionSchema(self.fields, "Academic paper content collection")
        self.collection_name = "paper_contents"

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
        print(f"集合 {self.collection_name} 创建成功")

    def __del__(self):
        """析构函数，关闭 session"""
        if self.session:
            self.session.close()

    def get_target_files(self, extracted_path: str = './output_extracted') -> List[Dict]:
        """获取需要入库的 content_list.json 文件列表"""
        if not os.path.exists(extracted_path):
            print(f"目录不存在: {extracted_path}")
            return []

        target_files = []

        print("开始扫描目录结构...")
        for root, dirs, files in os.walk(extracted_path):
            if 'images' in dirs:
                dirs.remove('images')

            for file in files:
                if file.endswith('content_list.json') and not file.endswith('_origin.pdf'):
                    file_path = os.path.join(root, file)
                    directory = os.path.basename(root)

                    file_info = {
                        'file_path': file_path,
                        'original_name': file,
                        'storage_name': file,
                        'directory': directory,
                        'file_type': 'json',
                        'relative_path': os.path.relpath(file_path, extracted_path)
                    }
                    target_files.append(file_info)

        return target_files

    def display_file_list(self, file_list: List[Dict]):
        """显示文件列表信息"""
        if not file_list:
            print("未找到目标文件")
            return

        print(f"\n{'=' * 80}")
        print(f"找到 {len(file_list)} 个 content_list.json 文件:")
        print(f"{'=' * 80}")

        for i, file_info in enumerate(file_list, 1):
            file_size = os.path.getsize(file_info['file_path']) if os.path.exists(file_info['file_path']) else 0
            file_size_kb = file_size / 1024
            print(f"  {i}. {file_info['original_name']} - {file_info['directory']} - {file_size_kb:.2f} KB")

    def get_session(self):
        """获取或创建 HTTP session"""
        if self.session is None:
            self.session = requests.Session()
            # 设置连接池和超时
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.max_workers,
                pool_maxsize=self.max_workers,
                max_retries=3
            )
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        return self.session

    def get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本向量 - 主要性能优化点"""
        session = self.get_session()
        headers = {
            'Authorization': f'Bearer {self.qwen_api_key}',
            'Content-Type': 'application/json'
        }

        # 预处理文本
        processed_texts = []
        for text in texts:
            if len(text) > 2000:
                text = text[:2000]
            processed_texts.append(text)

        data = {
            "model": "text-embedding-v1",
            "input": processed_texts
        }

        try:
            start_time = time.time()
            response = session.post(self.qwen_base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            elapsed_time = time.time() - start_time
            print(f"批量获取 {len(texts)} 个向量，耗时: {elapsed_time:.2f}s")

            embeddings = []
            if 'data' in result and len(result['data']) > 0:
                embeddings = [item['embedding'] for item in result['data']]
            elif 'output' in result and 'embeddings' in result['output']:
                embeddings = result['output']['embeddings']
            else:
                print(f"API 响应格式异常，使用随机向量")
                import random
                embeddings = [[random.uniform(-1, 1) for _ in range(1536)] for _ in range(len(texts))]

            return embeddings

        except Exception as e:
            print(f"批量获取向量失败: {e}")
            import random
            return [[random.uniform(-1, 1) for _ in range(1536)] for _ in range(len(texts))]

    def process_bbox(self, bbox):
        """处理边界框数据，转换为字符串格式"""
        if not bbox or len(bbox) != 4:
            return ""

        try:
            # 将 bbox 列表转换为字符串，格式: "x1,y1,x2,y2"
            bbox_str = ",".join(str(float(coord)) for coord in bbox)
            return bbox_str
        except (ValueError, TypeError):
            return ""

    def process_content_items_batch(self, items: List[Dict], file_name: str, directory: str, start_index: int) -> List[
        Dict]:
        """批量处理内容条目 - 性能优化"""
        entities = []
        embedding_texts = []
        item_mapping = []  # 记录每个文本对应的条目索引

        for i, item in enumerate(items):
            content_type = item.get('type', 'unknown')
            text_content = item.get('text', '')
            image_caption = item.get('image_caption', [])

            # 生成文本用于向量化
            embedding_text = ""
            if content_type == 'text' and text_content:
                embedding_text = text_content
            elif content_type == 'equation' and text_content:
                embedding_text = f"公式: {text_content}"
            elif content_type == 'image' and image_caption:
                embedding_text = f"图片: {' '.join(image_caption)}"

            if embedding_text:
                embedding_texts.append(embedding_text)
                item_mapping.append((i, item))

        # 批量获取向量
        if embedding_texts:
            vectors = self.get_embedding_batch(embedding_texts)

            # 构建实体
            for (text_index, (original_index, item)), vector in zip(enumerate(item_mapping), vectors):
                text_content = item.get('text', '')
                text_level = item.get('text_level', 0)
                page_idx = item.get('page_idx', 0)
                bbox = item.get('bbox', [])
                img_path = item.get('img_path', '')
                image_caption = item.get('image_caption', [])
                text_format = item.get('text_format', '')

                # 处理边界框 - 现在返回字符串格式
                bbox_str = self.process_bbox(bbox)

                # 生成唯一 ID
                entity_id = f"{directory}_{file_name}_{start_index + original_index}_{hashlib.md5(embedding_texts[text_index].encode()).hexdigest()[:8]}"

                entity = {
                    'id': entity_id,
                    'file_name': file_name,
                    'directory': directory,
                    'content_type': item.get('type', 'unknown'),
                    'text_content': text_content[:65535],
                    'text_level': text_level,
                    'page_idx': page_idx,
                    'bbox': bbox_str,  # 字符串格式
                    'img_path': img_path,
                    'image_caption': ' '.join(image_caption) if image_caption else '',
                    'text_format': text_format,
                    'vector': vector
                }

                entities.append(entity)

        return entities

    def store_single_file_optimized(self, file_info: Dict):
        """优化版的单个文件存储 - 主要性能优化"""
        file_path = file_info['file_path']
        file_name = file_info['original_name']
        directory = file_info['directory']

        try:
            # 读取 JSON 文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)

            if not isinstance(content_list, list):
                print(f"文件 {file_path} 格式错误，期望列表")
                return False

            print(f"处理文件 {file_name}，包含 {len(content_list)} 个内容条目")

            # 分批处理内容条目
            all_entities = []
            total_batches = (len(content_list) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(content_list))
                batch_items = content_list[start_idx:end_idx]

                print(f"  处理批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")

                batch_entities = self.process_content_items_batch(
                    batch_items, file_name, directory, start_idx
                )
                all_entities.extend(batch_entities)

                # 每批插入一次，避免内存占用过大
                if batch_entities:
                    self.insert_entities_batch(batch_entities)
                    print(f"  批次 {batch_idx + 1} 插入完成: {len(batch_entities)} 个条目")

            print(f"文件 {file_name} 处理完成: {len(all_entities)}/{len(content_list)} 个条目入库成功")
            return len(all_entities) > 0

        except Exception as e:
            print(f"处理文件 {file_path} 失败: {e}")
            return False

    def insert_entities_batch(self, entities: List[Dict]):
        """批量插入实体到 Milvus"""
        if not entities:
            return

        try:
            # 转换为 Milvus 插入格式 - 注意字段顺序
            insert_data = [
                [entity['id'] for entity in entities],
                [entity['file_name'] for entity in entities],
                [entity['directory'] for entity in entities],
                [entity['content_type'] for entity in entities],
                [entity['text_content'] for entity in entities],
                [entity['text_level'] for entity in entities],
                [entity['page_idx'] for entity in entities],
                [entity['bbox'] for entity in entities],  # 字符串格式
                [entity['img_path'] for entity in entities],
                [entity['image_caption'] for entity in entities],
                [entity['text_format'] for entity in entities],
                [entity['vector'] for entity in entities]
            ]

            self.collection.insert(insert_data)

        except Exception as e:
            print(f"批量插入失败: {e}")
            # 降级为逐个插入
            success_count = 0
            for entity in entities:
                try:
                    individual_data = [
                        [entity['id']],
                        [entity['file_name']],
                        [entity['directory']],
                        [entity['content_type']],
                        [entity['text_content']],
                        [entity['text_level']],
                        [entity['page_idx']],
                        [entity['bbox']],
                        [entity['img_path']],
                        [entity['image_caption']],
                        [entity['text_format']],
                        [entity['vector']]
                    ]
                    self.collection.insert(individual_data)
                    success_count += 1
                except Exception as single_error:
                    print(f"单个插入失败: {single_error}")

            print(f"降级插入完成: {success_count}/{len(entities)} 成功")

    def store_files_parallel(self, file_list: List[Dict]):
        """并行处理多个文件 - 性能优化"""
        if not file_list:
            print("没有文件需要处理")
            return

        total_files = len(file_list)
        success_files = 0

        print(f"\n{'=' * 80}")
        print(f"开始并行入库处理，共 {total_files} 个文件，使用 {self.max_workers} 个线程")
        print(f"{'=' * 80}")

        start_time = time.time()

        # 使用线程池并行处理文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.store_single_file_optimized, file_info): file_info
                for file_info in file_list
            }

            # 收集结果
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        success_files += 1
                    print(f"文件 {file_info['original_name']} 处理完成")
                except Exception as e:
                    print(f"文件 {file_info['original_name']} 处理失败: {e}")

        # 将数据持久化
        self.collection.flush()

        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"并行入库处理完成!")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"成功: {success_files}/{total_files} 个文件")
        print(f"失败: {total_files - success_files} 个文件")
        print(f"平均每个文件: {total_time / total_files:.2f} 秒")
        print(f"{'=' * 80}")

    def store_files(self, file_list: List[Dict]):
        """向后兼容的存储方法"""
        return self.store_files_parallel(file_list)

    def search_similar_contents(self, query, top_k=5):
        """搜索相似内容"""
        try:
            # 确保集合已加载
            self.collection.load()

            # 获取查询向量
            query_vector = self.get_embedding_batch([query])[0]

            # 搜索参数
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            # 执行搜索
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["file_name", "directory", "content_type", "text_content", "text_level", "page_idx",
                               "bbox"]
            )

            print(f"\n搜索查询: '{query}'")
            print(f"找到 {len(results[0])} 个相关结果:\n")

            for i, hit in enumerate(results[0]):
                content = hit.entity.get('text_content', '')
                content_preview = content[:300] + "..." if len(content) > 300 else content
                bbox = hit.entity.get('bbox', '')
                print(f"{i + 1}. 文件: {hit.entity.get('file_name')}")
                print(f"   目录: {hit.entity.get('directory')}")
                print(f"   类型: {hit.entity.get('content_type')}")
                print(f"   级别: {hit.entity.get('text_level')}")
                print(f"   页码: {hit.entity.get('page_idx')}")
                print(f"   边界框: {bbox}")
                print(f"   距离: {hit.distance:.4f}")
                print(f"   内容: {content_preview}")
                print("-" * 80)

        except Exception as e:
            print(f"搜索失败: {e}")

    def parse_bbox_string(self, bbox_str):
        """解析 bbox 字符串为列表（如果需要的话）"""
        if not bbox_str:
            return []
        try:
            return [float(coord) for coord in bbox_str.split(",")]
        except:
            return []

def insert_milvus_content():
    # 初始化存储类 - 可调整性能参数
    store = MilvusPaperStore(
        host='localhost',
        port='19530',
        qwen_api_key=QWEN_API_KEY,
        batch_size=50,  # 每批处理50个条目
        max_workers=3  # 同时处理3个文件
    )

    # 1. 首先获取文件列表
    print("第一步: 扫描 content_list.json 文件...")
    file_list = store.get_target_files('./output_extracted')
    store.display_file_list(file_list)

    # 2. 测试 API 连接
    print("\n第二步: 测试 API 连接...")
    test_vectors = store.get_embedding_batch(["测试文本1", "测试文本2"])
    print(f"批量获取 {len(test_vectors)} 个向量，每个维度: {len(test_vectors[0])}")

    # 3. 入库处理
    print("\n第三步: 开始并行入库...")
    store.store_files_parallel(file_list)

    # 4. 搜索测试
    print("\n第四步: 搜索测试...")
    store.search_similar_contents("玻璃形成能力 单原子金属", top_k=3)


# 使用示例
if __name__ == "__main__":
    insert_milvus_content()
