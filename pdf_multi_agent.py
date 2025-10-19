import os
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Vector store imports
from pymilvus import connections, Collection, utility

from config import DEEPSEEK_API_KEY


class AgentType(Enum):
    """智能体类型枚举"""
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"  # 知识库检索智能体
    DOMAIN_ADVISOR = "domain_advisor"  # 领域顾问智能体
    LEARNING_CONSULTANT = "learning_consultant"  # 学习顾问智能体


@dataclass
class SearchResult:
    """统一搜索结果数据结构"""
    rank: int
    score: float
    content: str
    metadata: Dict
    source_type: str


class VectorStoreManager:
    """知识库检索管理器 - 适配Milvus向量库"""

    def __init__(self, host='localhost', port='19530', api_key=None):
        self.host = host
        self.port = port
        self.api_key = api_key or os.getenv('QWEN_API_KEY')
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings'

        # 向量库集合配置
        self.collections = {
            'paper_contents': 'paper_contents',  # 论文内容
            'paper_figures': 'paper_figures',  # 论文图片
            'file_contents': 'file_contents'  # 文件内容
        }

        self._connect_milvus()

    def _connect_milvus(self):
        """连接Milvus数据库"""
        try:
            connections.connect(host=self.host, port=self.port)
            print("✅ Milvus知识库连接成功")
        except Exception as e:
            print(f"❌ Milvus连接失败: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        data = {"model": "text-embedding-v1", "input": text[:2000]}

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            result = response.json()

            if 'data' in result and result['data']:
                return result['data'][0]['embedding']
            else:
                import random
                return [random.uniform(-1, 1) for _ in range(1536)]

        except Exception as e:
            print(f"获取向量失败: {e}")
            import random
            return [random.uniform(-1, 1) for _ in range(1536)]

    def search_papers(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """论文检索"""
        return self._search_collection(
            self.collections['paper_contents'], query, top_k,
            ["file_name", "directory", "content_type", "text_content",
             "text_level", "page_idx", "text_format", "image_caption"]
        )

    def search_content(self, query: str, top_k: int = 8) -> List[SearchResult]:
        """论文内容检索"""
        results = self._search_collection(
            self.collections['paper_contents'], query, top_k,
            ["file_name", "directory", "content_type", "text_content",
             "text_level", "page_idx", "text_format"]
        )
        return [r for r in results if r.metadata.get('content_type') == 'text']

    def search_figures(self, query: str, top_k: int = 6) -> List[SearchResult]:
        """论文图片检索"""
        return self._search_collection(
            self.collections['paper_figures'], query, top_k,
            ["file_name", "figure_id", "figure_name", "section",
             "conclusion", "subfigure_count", "subfigure_data", "content"]
        )

    def _search_collection(self, collection_name: str, query: str, top_k: int,
                           output_fields: List[str]) -> List[SearchResult]:
        """通用向量搜索方法"""
        if not utility.has_collection(collection_name):
            print(f"❌ 集合 {collection_name} 不存在")
            return []

        collection = Collection(collection_name)
        collection.load()

        query_vector = self.get_embedding(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        try:
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )

            formatted_results = []
            for i, hit in enumerate(results[0]):
                result = SearchResult(
                    rank=i + 1,
                    score=1 - float(hit.distance),
                    content=self._extract_content(hit.entity, collection_name),
                    metadata={field: hit.entity.get(field) for field in output_fields},
                    source_type=collection_name
                )
                formatted_results.append(result)

            return formatted_results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def _extract_content(self, entity, collection_name: str) -> str:
        """根据集合类型提取主要内容"""
        if collection_name == self.collections['paper_contents']:
            return entity.get('text_content', '')
        elif collection_name == self.collections['paper_figures']:
            figure_name = entity.get('figure_name', '')
            conclusion = entity.get('conclusion', '')
            return f"{figure_name}: {conclusion}"
        elif collection_name == self.collections['file_contents']:
            return entity.get('content', '')
        return ""


class LLMService:
    """统一的LLM服务"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.7
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """生成回复"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"❌ API调用失败: {str(e)}"


class BaseAgent:
    """智能体基类"""

    def __init__(self):
        self.llm = LLMService()
        self.vector_store = VectorStoreManager()


class KnowledgeRetrieverAgent(BaseAgent):
    """知识库检索智能体 - 负责所有检索功能"""

    def retrieve_information(self, query: str, search_type: str = "all") -> str:
        """综合检索信息"""
        print(f"🔍 知识库检索: {query} - 类型: {search_type}")

        if search_type == "papers" or search_type == "all":
            papers = self.vector_store.search_papers(query, top_k=8)
        else:
            papers = []

        if search_type == "content" or search_type == "all":
            contents = self.vector_store.search_content(query, top_k=6)
        else:
            contents = []

        if search_type == "figures" or search_type == "all":
            figures = self.vector_store.search_figures(query, top_k=5)
        else:
            figures = []

        prompt = f"""
        # 知识库检索报告

        ## 检索查询
        {query}

        ## 检索类型
        {search_type}

        ## 检索结果

        ### 📚 相关论文 ({len(papers)}篇)
        {self._format_paper_results(papers)}

        ### 📖 相关内容 ({len(contents)}条)
        {self._format_content_results(contents)}

        ### 🖼️ 相关图表 ({len(figures)}个)
        {self._format_figure_results(figures)}

        请基于以上检索结果：
        1. 总结检索到的主要信息
        2. 分析信息的相关性和完整性
        3. 提供进一步检索的建议

        要求：全面客观，重点突出检索结果。
        """

        system_prompt = "你是知识库检索专家，擅长从向量数据库中高效检索相关信息。"
        return self.llm.generate_response(prompt, system_prompt)

    def _format_paper_results(self, papers: List[SearchResult]) -> str:
        """格式化论文结果"""
        if not papers:
            return "未找到相关论文"

        formatted = ""
        file_groups = {}
        for paper in papers:
            filename = paper.metadata.get('file_name', '未知文件')
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append(paper)

        for filename, papers_in_file in list(file_groups.items())[:5]:
            formatted += f"**{filename}**\n"
            for paper in papers_in_file[:2]:
                content_preview = paper.content[:150] + "..." if len(paper.content) > 150 else paper.content
                formatted += f"- {content_preview} (相关度: {paper.score:.3f})\n"
            formatted += "\n"

        return formatted

    def _format_content_results(self, contents: List[SearchResult]) -> str:
        """格式化内容结果"""
        if not contents:
            return "未找到相关内容"

        formatted = ""
        for content in contents[:5]:
            filename = content.metadata.get('file_name', '未知文件')
            page = content.metadata.get('page_idx', 0) + 1
            text = content.content[:200] + "..." if len(content.content) > 200 else content.content
            formatted += f"**{filename}** (第{page}页, 相关度: {content.score:.3f})\n"
            formatted += f"{text}\n\n"

        return formatted

    def _format_figure_results(self, figures: List[SearchResult]) -> str:
        """格式化图表结果"""
        if not figures:
            return "未找到相关图表"

        formatted = ""
        for figure in figures:
            name = figure.metadata.get('figure_name', '未知图表')
            conclusion = figure.metadata.get('conclusion', '')
            section = figure.metadata.get('section', '')

            formatted += f"### {name}\n"
            if section:
                formatted += f"**章节**: {section}\n"
            formatted += f"**结论**: {conclusion}\n"
            formatted += f"**相关度**: {figure.score:.3f}\n\n"

        return formatted


class DomainAdvisorAgent(BaseAgent):
    """领域顾问智能体 - 负责专业分析和建议"""

    def provide_expert_advice(self, query: str, advice_type: str = "comprehensive") -> str:
        """提供专业领域建议"""
        print(f"🎓 领域建议: {query} - 类型: {advice_type}")

        # 搜索相关基础信息
        papers = self.vector_store.search_papers(query, top_k=6)
        contents = self.vector_store.search_content(query, top_k=4)
        figures = self.vector_store.search_figures(query, top_k=3)

        prompt = f"""
        # 材料科学领域专家建议

        ## 咨询问题
        {query}

        ## 咨询类型
        {advice_type}

        ## 相关研究基础
        {self._format_expert_content(papers, contents, figures)}

        {self._get_analysis_framework(advice_type)}

        要求：分析专业深入，建议具体可行，基于科学原理。
        """

        system_prompt = """你是资深的材料科学领域专家，在金属材料、相变行为、晶体结构等方面有深厚造诣。
        你的分析应该基于科学原理，建议应该具体可行。"""

        return self.llm.generate_response(prompt, system_prompt)

    def _get_analysis_framework(self, advice_type: str) -> str:
        """根据建议类型获取分析框架"""
        frameworks = {
            "comprehensive": """
            请从以下维度进行全面分析：

            ### 🎯 核心内容提炼
            - 主要科学问题和研究方法
            - 关键实验发现和技术创新
            - 重要理论贡献

            ### 💎 研究价值评估
            - 理论创新性和科学意义
            - 技术突破和应用前景
            - 对领域发展的影响

            ### 🔬 专业概念解析
            - 相关专业概念的准确解释
            - 物理/化学机制分析
            - 在材料科学中的具体表现

            ### 📊 实验数据解读
            - 关键数据的科学含义
            - 实验结果的可靠性分析
            - 数据趋势的物理意义

            ### 🛠️ 方法技术指导
            - 实验设计和技术路线
            - 操作要点和注意事项
            - 替代方案比较

            ### ⚖️ 性能系统比较
            - 关键性能参数对比
            - 优缺点分析
            - 适用场景评估

            ### 🚀 优化发展建议
            - 研究方向建议
            - 技术改进方案
            - 创新机会识别
            """,

            "research": """
            请重点进行研究方向分析：

            ### 🔍 研究现状分析
            - 当前研究水平和进展
            - 主要技术路线和方法
            - 存在的科学问题

            ### 🎯 研究方向建议
            - 潜在的突破方向
            - 交叉学科机会
            - 前沿热点追踪

            ### 💡 优化改进建议
            - 实验设计优化
            - 数据分析方法改进
            - 理论模型完善
            """,

            "concept": """
            请重点进行概念解析：

            ### 📖 概念定义阐述
            - 准确的定义和内涵
            - 相关术语说明
            - 历史发展脉络

            ### 🎯 科学原理分析
            - 物理/化学机制
            - 理论基础和模型
            - 关键参数和特征

            ### 💡 应用意义说明
            - 在材料科学中的重要性
            - 实际应用场景
            - 研究价值评估
            """,

            "method": """
            请重点进行方法指导：

            ### 🛠️ 技术路线设计
            - 实验流程规划
            - 关键技术选择
            - 质量控制措施

            ### ⚙️ 操作实施指南
            - 具体操作步骤
            - 仪器设备要求
            - 参数设置建议

            ### 📊 数据分析方法
            - 数据处理流程
            - 统计分析方法
            - 结果验证方案
            """
        }

        return frameworks.get(advice_type, frameworks["comprehensive"])

    def _format_expert_content(self, papers: List[SearchResult], contents: List[SearchResult],
                               figures: List[SearchResult]) -> str:
        """格式化专家分析内容"""
        formatted = "### 相关研究内容\n\n"

        if papers:
            formatted += "#### 论文研究成果\n"
            for paper in papers[:4]:
                content_preview = paper.content[:250] + "..." if len(paper.content) > 250 else paper.content
                formatted += f"- {content_preview} (相关度: {paper.score:.3f})\n"
            formatted += "\n"

        if contents:
            formatted += "#### 具体内容分析\n"
            for content in contents[:3]:
                formatted += f"- {content.content[:200]}...\n"
            formatted += "\n"

        if figures:
            formatted += "#### 实验数据支撑\n"
            for figure in figures[:2]:
                name = figure.metadata.get('figure_name', '未知图表')
                conclusion = figure.metadata.get('conclusion', '')[:150] + "..." if len(
                    figure.metadata.get('conclusion', '')) > 150 else figure.metadata.get('conclusion', '')
                formatted += f"- {name}: {conclusion} (相关度: {figure.score:.3f})\n"

        return formatted


class LearningConsultantAgent(BaseAgent):
    """学习顾问智能体 - 负责学习建议和资料推荐"""

    def provide_learning_guidance(self, user_input: str) -> str:
        """提供学习指导和建议"""
        print(f"🎓 学习指导: {user_input}")

        # 搜索相关知识内容用于分析基础
        papers = self.vector_store.search_papers(user_input, top_k=4)
        contents = self.vector_store.search_content(user_input, top_k=3)

        prompt = f"""
        # 个性化学习指导方案

        ## 学习需求分析
        {user_input}

        ## 现有知识基础
        {self._format_learning_base(papers, contents)}

        请基于以上信息，提供全面的学习指导：

        ### 🔍 知识盲点诊断
        - 可能存在的理解误区分析
        - 基础概念的掌握程度评估
        - 技术细节的认知差距识别

        ### 🌐 最优学习资料推荐

        #### 网络学习资源
        - 优质在线课程和MOOC平台推荐
        - 专业学习网站和数据库
        - 最新研究动态和学术资讯来源

        #### 经典学术文献
        - 必读的基础理论教科书
        - 重要的综述性论文
        - 前沿研究突破性文章

        #### 实践学习材料
        - 实验操作手册和指南
        - 仿真模拟工具和教程
        - 数据分析软件和学习资源

        ### 🗺️ 系统学习路径

        #### 初级阶段
        - 基础理论知识学习
        - 核心概念理解
        - 基本技能培养

        #### 进阶阶段  
        - 专业深度知识学习
        - 研究方法掌握
        - 实践能力提升

        #### 高级阶段
        - 前沿技术学习
        - 创新能力培养
        - 学术交流参与

        ### 💡 高效学习策略
        - 个性化学习方法建议
        - 知识巩固和实践技巧
        - 学习进度管理和评估

        ### 🎯 学习目标设定
        - 短期目标（1-3个月）
        - 中期目标（6-12个月） 
        - 长期目标（1-2年）

        要求：分析准确深入，推荐具体实用，路径清晰可行。
        """

        system_prompt = """你是专业的学术导师和学习顾问，擅长分析学生的学习需求并提供最优的学习方案。
        你的建议应该基于最新的教育资源和学习方法，提供个性化的成长路径。"""

        return self.llm.generate_response(prompt, system_prompt)

    def _format_learning_base(self, papers: List[SearchResult], contents: List[SearchResult]) -> str:
        """格式化学习基础内容"""
        formatted = "### 相关知识内容概览\n\n"

        if papers or contents:
            all_contents = list(papers) + list(contents)
            all_contents.sort(key=lambda x: x.score, reverse=True)

            for item in all_contents[:5]:
                content_preview = item.content[:180] + "..." if len(item.content) > 180 else item.content
                formatted += f"- {content_preview} (相关度: {item.score:.3f})\n"
        else:
            formatted += "当前知识库中相关内容有限，将主要基于通用学习理论提供建议\n"

        formatted += "\n### 💡 提示：学习建议将结合最新网络资源和学术动态"

        return formatted


class MaterialScienceQASystem:
    """材料科学问答系统 - 三大核心智能体"""

    def __init__(self):
        # 初始化三大核心智能体
        self.agents = {
            AgentType.KNOWLEDGE_RETRIEVER: KnowledgeRetrieverAgent(),
            AgentType.DOMAIN_ADVISOR: DomainAdvisorAgent(),
            AgentType.LEARNING_CONSULTANT: LearningConsultantAgent(),
        }

        # 智能体路由配置
        self.agent_routing = {
            AgentType.KNOWLEDGE_RETRIEVER: [
                '检索', '搜索', '查找', '论文检索', '内容检索', '图片检索',
                'retrieve', 'search', 'find', 'paper search', 'content search'
            ],
            AgentType.DOMAIN_ADVISOR: [
                '分析', '评估', '建议', '解释', '指导', '比较', '研究方向',
                'analyze', 'evaluate', 'advice', 'explain', 'guidance', 'comparison'
            ],
            AgentType.LEARNING_CONSULTANT: [
                '学习', '资料', '盲点', '路径', '建议', '如何学习', '学习计划',
                'learn', 'study', 'materials', 'blind spot', 'learning path'
            ]
        }

    def route_query(self, query: str) -> Tuple[AgentType, float]:
        """路由查询到合适的智能体"""
        query_lower = query.lower()

        agent_scores = {}
        for agent_type, keywords in self.agent_routing.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            agent_scores[agent_type] = score

        best_agent = max(agent_scores, key=agent_scores.get)
        confidence = agent_scores[best_agent] / max(len(self.agent_routing[best_agent]), 1)

        return best_agent, confidence

    def process_query(self, query: str) -> str:
        """处理用户查询"""
        agent_type, confidence = self.route_query(query)
        print(f"🤖 分配智能体: {agent_type.value} (置信度: {confidence:.2f})")

        agent = self.agents[agent_type]

        try:
            if agent_type == AgentType.KNOWLEDGE_RETRIEVER:
                # 根据查询内容确定检索类型
                if '论文' in query or '文献' in query:
                    search_type = "papers"
                elif '内容' in query or '文本' in query:
                    search_type = "content"
                elif '图片' in query or '图表' in query or '图' in query:
                    search_type = "figures"
                else:
                    search_type = "all"
                return agent.retrieve_information(query, search_type)

            elif agent_type == AgentType.DOMAIN_ADVISOR:
                # 根据查询内容确定建议类型
                if '概念' in query or '解释' in query or '是什么' in query:
                    advice_type = "concept"
                elif '方法' in query or '技术' in query or '实验' in query:
                    advice_type = "method"
                elif '研究' in query or '方向' in query or '优化' in query:
                    advice_type = "research"
                else:
                    advice_type = "comprehensive"
                return agent.provide_expert_advice(query, advice_type)

            elif agent_type == AgentType.LEARNING_CONSULTANT:
                return agent.provide_learning_guidance(query)

            else:
                return "暂不支持该类型查询"

        except Exception as e:
            return f"❌ 处理查询时出错: {str(e)}"


def run_three_agents_tests():
    """运行三大智能体测试"""
    print("🔬 材料科学智能问答系统 - 三大智能体测试")
    print("=" * 60)

    # 设置API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

    qa_system = MaterialScienceQASystem()

    # 三大智能体测试用例
    test_cases = [
        # 知识库检索智能体测试
        {
            "query": "检索关于金属玻璃形成能力的相关论文",
            "expected_agent": AgentType.KNOWLEDGE_RETRIEVER,
            "description": "测试论文检索功能"
        },
        {
            "query": "搜索Ta和Zr结晶动力学差异的具体内容",
            "expected_agent": AgentType.KNOWLEDGE_RETRIEVER,
            "description": "测试内容检索功能"
        },
        {
            "query": "查找EAM势函数比较的图表",
            "expected_agent": AgentType.KNOWLEDGE_RETRIEVER,
            "description": "测试图片检索功能"
        },

        # 领域顾问智能体测试
        {
            "query": "分析胡远超老师论文的核心内容和研究价值",
            "expected_agent": AgentType.DOMAIN_ADVISOR,
            "description": "测试研究分析功能"
        },
        {
            "query": "解释什么是准晶相及其在金属玻璃中的作用",
            "expected_agent": AgentType.DOMAIN_ADVISOR,
            "description": "测试概念解释功能"
        },
        {
            "query": "指导如何用分子动力学研究金属相变",
            "expected_agent": AgentType.DOMAIN_ADVISOR,
            "description": "测试方法指导功能"
        },
        {
            "query": "比较Ta和Zr的玻璃形成能力",
            "expected_agent": AgentType.DOMAIN_ADVISOR,
            "description": "测试性能比较功能"
        },

        # 学习顾问智能体测试
        {
            "query": "我想学习金属玻璃研究，请分析知识盲点并推荐学习资料",
            "expected_agent": AgentType.LEARNING_CONSULTANT,
            "description": "测试学习建议功能"
        },
        {
            "query": "如何系统学习分子动力学模拟方法",
            "expected_agent": AgentType.LEARNING_CONSULTANT,
            "description": "测试学习路径规划"
        }
    ]

    print("🧪 开始运行三大智能体测试...")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {test_case['description']}")
        print(f"💬 查询: {test_case['query']}")
        print("-" * 50)

        assigned_agent, confidence = qa_system.route_query(test_case['query'])
        print(f"🎯 分配智能体: {assigned_agent.value}")
        print(f"📊 分配置信度: {confidence:.2f}")

        if assigned_agent == test_case['expected_agent']:
            print("✅ 智能体分配正确")
        else:
            print(f"⚠️  期望: {test_case['expected_agent'].value}")

        response = qa_system.process_query(test_case['query'])
        print(f"🤖 回答预览: {response[:300]}...")

        if i < len(test_cases):
            print("\n" + "=" * 60)

    print(f"\n🎉 三大智能体测试完成！")
    print("📈 系统架构:")
    print("  ✅ 知识库检索智能体 - 论文、内容、图片检索")
    print("  ✅ 领域顾问智能体 - 研究分析、概念解释、方法指导、数据解读、性能比较")
    print("  ✅ 学习顾问智能体 - 知识盲点分析、学习资料推荐、学习路径规划")


def pdf_agent():
    """主函数 - 增强版用户交互"""
    print("🔬 材料科学智能问答系统")
    print("=" * 60)
    print("🌟 三大核心智能体为您服务:")
    print("")
    print("  📚 知识检索 - 找论文、查内容、搜图表")
    print("  🎓 领域专家 - 深度分析、专业解释、方法指导")
    print("  🌐 学习顾问 - 学习规划、资料推荐、盲点分析")
    print("")
    print("💡 使用提示:")
    print("  • 输入 '帮助' 查看使用指南")
    print("  • 输入 '模式' 查看智能体功能说明")
    print("  • 输入 '退出' 结束对话")
    print("=" * 60)
    print("输入示例：检索金属玻璃形成的相关论文")

    # 设置API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

    # 初始化系统
    qa_system = MaterialScienceQASystem()
    print("✅ 系统初始化完成！")
    print("")

    conversation_history = []

    while True:
        try:
            user_input = input("💬 您的问题: ").strip()

            # 特殊命令处理
            if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
                print("👋 感谢使用，期待再次为您服务！")
                break

            elif user_input.lower() in ['帮助', 'help']:
                show_help()
                continue

            elif user_input.lower() in ['模式', '模式说明', 'agents']:
                show_agent_modes()
                continue

            elif user_input.lower() in ['历史', 'history']:
                show_conversation_history(conversation_history)
                continue

            elif user_input.lower() in ['清除', 'clear']:
                conversation_history.clear()
                print("🗑️  对话历史已清除")
                continue

            if not user_input:
                print("⚠️  请输入有效问题")
                continue

            print("⏳ 正在分析您的问题...")

            # 记录对话历史
            conversation_history.append({"role": "user", "content": user_input})

            # 处理查询
            agent_type, confidence = qa_system.route_query(user_input)
            response = qa_system.process_query(user_input)

            # 记录回复历史
            conversation_history.append({"role": "assistant", "content": response, "agent": agent_type.value})

            # 显示结果
            display_response(user_input, response, agent_type, confidence)

        except KeyboardInterrupt:
            print("\n\n🛑 对话结束，感谢使用！")
            break
        except Exception as e:
            error_msg = f"系统错误: {str(e)}"
            print(f"\n❌ {error_msg}")
            conversation_history.append({"role": "system", "content": error_msg})


def display_response(question: str, response: str, agent_type: AgentType, confidence: float):
    """美化显示回复结果"""
    print("\n" + "=" * 70)
    print(f"🎯 问题: {question}")
    print(f"🤖 处理智能体: {agent_type.value} (置信度: {confidence:.2f})")
    print("=" * 70)
    print(f"\n{response}")
    print("\n" + "=" * 70)
    print("💡 您可以继续提问，或输入'帮助'查看使用指南")
    print("")


def show_help():
    """显示帮助信息"""
    help_text = """
📖 使用指南:

【基本操作】
• 直接输入问题即可获得答案
• 输入 '退出' 结束对话
• 输入 '历史' 查看对话记录
• 输入 '清除' 清空对话历史

【智能体功能】
• 知识检索: 包含"检索"、"搜索"、"查找"等关键词
• 领域专家: 包含"分析"、"解释"、"指导"等关键词  
• 学习顾问: 包含"学习"、"资料"、"盲点"等关键词

【示例问题】
• "检索金属玻璃形成的相关论文"
• "解释什么是准晶相"
• "如何学习分子动力学模拟"
• "分析论文的研究价值"
"""
    print(help_text)


def show_agent_modes():
    """显示智能体模式说明"""
    modes_text = """
🤖 三大智能体功能详解:

📚 【知识库检索智能体】
  擅长: 文献检索、内容搜索、图表查找
  示例:
    • "检索Ta Zr金属玻璃的相关论文"
    • "搜索EAM势函数比较的图表" 
    • "查找相变动力学的具体内容"

🎓 【领域顾问智能体】  
  擅长: 专业分析、概念解释、方法指导
  示例:
    • "分析金属玻璃的形成能力"
    • "解释晶体缺陷对性能的影响"
    • "指导分子动力学模拟方法"
    • "比较不同材料的力学性能"

🌐 【学习顾问智能体】
  擅长: 学习规划、资料推荐、盲点分析
  示例:
    • "如何系统学习材料科学"
    • "推荐金属相变的学习资料"
    • "分析我在晶体学方面的知识盲点"
"""
    print(modes_text)


def show_conversation_history(history: list):
    """显示对话历史"""
    if not history:
        print("📝 对话历史为空")
        return

    print("\n📝 对话历史:")
    print("=" * 50)
    for i, item in enumerate(history, 1):
        if item["role"] == "user":
            print(f"👤 第{i}问: {item['content']}")
        elif item["role"] == "assistant":
            print(f"🤖 第{i}答: {item['content'][:100]}...")
            print(f"   智能体: {item.get('agent', '未知')}")
        print("-" * 30)
    print(f"总计: {len([h for h in history if h['role'] == 'user'])} 次问答")
    print("")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_three_agents_tests()
    else:
        pdf_agent()