import os
import requests
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import hashlib
import time
import uuid
from typing import List, Dict

from config import QWEN_API_KEY


class MilvusPaperStore:
    def __init__(self, host='localhost', port='19530', qwen_api_key=None, qwen_base_url=None):
        # 连接 Milvus
        connections.connect(host=host, port=port)

        # Qwen API 配置
        self.qwen_api_key = qwen_api_key or os.getenv('QWEN_API_KEY')
        self.qwen_base_url = qwen_base_url or os.getenv('QWEN_BASE_URL',
                                                        'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings')

        # 定义集合 schema - 增加安全边界
        self.fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="directory", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 最大 65535 字符
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
        ]

        self.schema = CollectionSchema(self.fields, "Academic paper collection")
        self.collection_name = "papers"

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

    def get_target_files(self, extracted_path: str = './output_extracted') -> List[Dict]:
        """获取需要入库的文件列表"""
        if not os.path.exists(extracted_path):
            print(f"目录不存在: {extracted_path}")
            return []

        target_files = []

        print("开始扫描目录结构...")
        for root, dirs, files in os.walk(extracted_path):
            if 'images' in dirs:
                dirs.remove('images')

            for file in files:
                if file == 'full.md' or (file.endswith('content_list.json') and not file.endswith('_origin.pdf')):
                    file_path = os.path.join(root, file)
                    directory = os.path.basename(root)

                    if file == 'full.md':
                        storage_file_name = f"{directory}_full.md"
                    else:
                        storage_file_name = f"{directory}_{file}"

                    file_info = {
                        'file_path': file_path,
                        'original_name': file,
                        'storage_name': storage_file_name,
                        'directory': directory,
                        'file_type': 'md' if file.endswith('.md') else 'json',
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
        print(f"找到 {len(file_list)} 个需要入库的文件:")
        print(f"{'=' * 80}")

        dir_groups = {}
        for file_info in file_list:
            dir_name = file_info['directory']
            if dir_name not in dir_groups:
                dir_groups[dir_name] = []
            dir_groups[dir_name].append(file_info)

        for dir_name, files in dir_groups.items():
            print(f"\n📁 目录: {dir_name}")
            for i, file_info in enumerate(files, 1):
                file_size = os.path.getsize(file_info['file_path']) if os.path.exists(file_info['file_path']) else 0
                file_size_mb = file_size / (1024 * 1024)
                print(
                    f"  {i}. {file_info['original_name']} → {file_info['storage_name']} ({file_info['file_type']}) - {file_size_mb:.2f} MB")

    def get_embedding_with_qwen(self, text):
        """使用 Qwen API 获取文本向量"""
        headers = {
            'Authorization': f'Bearer {self.qwen_api_key}',
            'Content-Type': 'application/json'
        }

        # 如果文本太长，进行截断（Qwen 限制大约 2000 字符）
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
            elif 'output' in result and 'embeddings' in result['output']:
                return result['output']['embeddings'][0]['embedding']
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

    def split_content(self, content, max_length=60000):
        """
        将长内容分割为多个片段，确保每个片段不超过最大长度
        使用更保守的分割策略，留出安全边界
        """
        # 设置安全边界，确保不超过 65535
        safe_max_length = min(max_length, 65000)

        if len(content) <= safe_max_length:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # 计算当前片段的结束位置（考虑安全边界）
            end = start + safe_max_length

            # 如果剩余内容不足 safe_max_length，直接取剩余部分
            if end >= len(content):
                chunks.append(content[start:])
                break

            # 优先在段落边界分割
            paragraph_break = content.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + safe_max_length * 0.3:  # 至少保留30%内容
                end = paragraph_break
            else:
                # 其次在句子边界分割
                sentence_break = content.rfind('。', start, end)
                if sentence_break != -1 and sentence_break > start + safe_max_length * 0.3:
                    end = sentence_break + 1  # 包括句号
                else:
                    # 最后在换行符处分割
                    line_break = content.rfind('\n', start, end)
                    if line_break != -1 and line_break > start + safe_max_length * 0.3:
                        end = line_break
                    else:
                        # 强制在安全位置分割
                        end = start + safe_max_length

            chunk = content[start:end].strip()
            if chunk:  # 确保片段不为空
                chunks.append(chunk)

            start = end

            # 跳过分割符
            while start < len(content) and content[start] in ['\n', ' ', '\t']:
                start += 1

        return chunks

    def validate_content_length(self, content_chunks):
        """验证所有内容片段都不超过长度限制"""
        validated_chunks = []
        for i, chunk in enumerate(content_chunks):
            if len(chunk) > 65535:
                print(f"警告: 片段 {i + 1} 仍然过长 ({len(chunk)} 字符)，进行强制截断")
                # 强制截断到安全长度
                chunk = chunk[:65000]
            validated_chunks.append(chunk)
        return validated_chunks

    def generate_entity_id(self, file_name, content_hash, chunk_index, total_chunks):
        """生成符合长度限制的实体 ID"""
        base_id = str(uuid.uuid4())
        if total_chunks > 1:
            return f"{base_id}_{chunk_index}"
        else:
            return base_id

    def store_single_file(self, file_info: Dict):
        """存储单个文件"""
        file_path = file_info['file_path']
        storage_name = file_info['storage_name']
        original_name = file_info['original_name']
        directory = file_info['directory']
        file_type = file_info['file_type']

        try:
            # 读取文件内容
            if file_type == 'md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)
                content = json.dumps(json_content, ensure_ascii=False, indent=2)
            else:
                print(f"不支持的文件类型: {file_path}")
                return False

            # 生成内容哈希
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]

            # 检查内容长度，如果超过限制则分割
            max_content_length = 65535
            if len(content) > max_content_length:
                print(f"内容过长 ({len(content)} 字符)，进行分割...")
                content_chunks = self.split_content(content, 60000)  # 使用更保守的初始分割长度
                content_chunks = self.validate_content_length(content_chunks)  # 验证长度
                print(f"分割为 {len(content_chunks)} 个片段")
            else:
                content_chunks = [content]

            # 存储每个片段
            success_count = 0
            for i, chunk in enumerate(content_chunks):
                # 最终长度检查
                if len(chunk) > max_content_length:
                    print(f"错误: 片段 {i + 1} 仍然过长 ({len(chunk)} 字符)，跳过此片段")
                    continue

                # 生成唯一 ID
                entity_id = self.generate_entity_id(storage_name, content_hash, i, len(content_chunks))

                # 获取向量
                print(f"正在为 {storage_name} 片段 {i + 1}/{len(content_chunks)} 生成向量...")
                vector = self.get_embedding_with_qwen(chunk[:2000])

                # 准备数据
                chunk_file_name = storage_name + (f"_part{i + 1}" if len(content_chunks) > 1 else "")

                data = [
                    [entity_id],
                    [chunk_file_name],
                    [file_type],
                    [directory],
                    [chunk],
                    [vector]
                ]

                # 插入数据
                try:
                    self.collection.insert(data)
                    success_count += 1
                    print(f"文件片段 {i + 1}/{len(content_chunks)} 存储成功 (长度: {len(chunk)} 字符)")
                except Exception as insert_error:
                    print(f"插入数据失败: {insert_error}")
                    # 尝试使用更短的 ID
                    short_entity_id = str(uuid.uuid4())[:32]
                    data[0] = [short_entity_id]
                    try:
                        self.collection.insert(data)
                        success_count += 1
                        print(f"文件片段 {i + 1}/{len(content_chunks)} 使用短ID存储成功")
                    except Exception as retry_error:
                        print(f"重试插入也失败: {retry_error}")

            print(
                f"文件 {original_name} → {storage_name} 处理完成: {success_count}/{len(content_chunks)} 个片段入库成功")
            return success_count > 0

        except Exception as e:
            print(f"存储文件 {file_path} 失败: {e}")
            return False

    def store_files(self, file_list: List[Dict]):
        """批量存储文件列表"""
        if not file_list:
            print("没有文件需要处理")
            return

        total_files = len(file_list)
        success_files = 0

        print(f"\n{'=' * 80}")
        print(f"开始入库处理，共 {total_files} 个文件")
        print(f"{'=' * 80}")

        for i, file_info in enumerate(file_list, 1):
            print(f"\n[{i}/{total_files}] 处理文件: {file_info['file_path']}")
            print(f"    存储名称: {file_info['storage_name']}")

            if self.store_single_file(file_info):
                success_files += 1

            time.sleep(1)  # 避免 API 限流

        # 将数据持久化
        self.collection.flush()

        print(f"\n{'=' * 80}")
        print(f"入库处理完成!")
        print(f"成功: {success_files}/{total_files} 个文件")
        print(f"失败: {total_files - success_files} 个文件")
        print(f"{'=' * 80}")

    def search_similar_papers(self, query, top_k=5):
        """搜索相似论文"""
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
                output_fields=["file_name", "file_type", "directory", "content"]
            )

            print(f"\n搜索查询: '{query}'")
            print(f"找到 {len(results[0])} 个相关结果:\n")

            for i, hit in enumerate(results[0]):
                content = hit.entity.get('content', '')
                content_preview = content[:200] + "..." if len(content) > 200 else content
                print(f"{i + 1}. 文件: {hit.entity.get('file_name')}")
                print(f"   类型: {hit.entity.get('file_type')}")
                print(f"   目录: {hit.entity.get('directory')}")
                print(f"   距离: {hit.distance:.4f}")
                print(f"   内容预览: {content_preview}")
                print()

        except Exception as e:
            print(f"搜索失败: {e}")

def insert_milvus_paper():
    # 初始化存储类
    store = MilvusPaperStore(
        host='localhost',
        port='19530',
        qwen_api_key=QWEN_API_KEY
    )

    # 1. 首先获取文件列表
    print("第一步: 扫描文件...")
    file_list = store.get_target_files('./output_extracted')
    store.display_file_list(file_list)

    # 2. 测试 API 连接
    print("\n第二步: 测试 API 连接...")
    test_vector = store.get_embedding_with_qwen("测试文本")
    print(f"向量维度: {len(test_vector)}")

    # 3. 入库处理
    print("\n第三步: 开始入库...")
    store.store_files(file_list)

    # 4. 搜索测试
    print("\n第四步: 搜索测试...")
    store.search_similar_papers("玻璃形成能力 单原子金属", top_k=3)


# 使用示例
if __name__ == "__main__":
    insert_milvus_paper()