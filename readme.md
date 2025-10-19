🔬 材料科学智能问答系统 - 三大核心智能体
============================================================
🏛️  系统架构:

  1. 📚 知识库检索智能体
     • 论文检索
     • 论文内容检索
     • 论文图片检索

  2. 🎓 领域顾问智能体(优化中...)
     • 论文核心内容提炼
     • 研究价值评估
     • 研究方向建议
     • 专业概念解释
     • 图表数据解读
     • 实验方法指导
     • 材料性能比较

  3. 🌐 学习顾问智能体(Todo)
     • 知识盲点分析
     • 最优学习资料搜寻
     • 个性化学习建议

============================================================


# 前置准备
### 1.安装依赖库
```bash
pip install -r requirements.txt
```
### 2.安装启动lilvus(建议用docker compose安装启动，file文件夹中有yml文件)


# 使用方式
## 直接运行
```bash
python main.py
```
或

## 分步执行（解析pdf->导入到milvus->启动agent系统）
### 解析pdf
#### 1.将需要解析的文件放入pdf_file中
#### 2.在config.py中修改MinerU token和milvus配置
#### 3.运行：
```bash
python parser_by_MinerU.py
```
### 将解析后的json、markdown和图片信息导入到milvus中
#### 1.启动milvus(需要先安装docker和docker compose)
```bash
cd file
docker-compose -f milvus-standalone-docker-compose.yml up -d
```
#### 2.按需将数据导入到milvus中
  ##### 2.1论文分段导入：
  ```bash
  python insert_milvus_paper.py
  ```
  ##### 2.2论文content导入：
  ```bash
  python insert_milvus_content.py
  ```
  ##### 2.3论文图片信息导入：
  ```bash
  python insert_milvus_figure.py
  ```
### 运行multi-agent对话系统
```bash
python pdf_multi_agent.py
```
### 使用样例：
```bash
1.检索“通过竞争有序平衡实现单质金属的玻璃形成”的相关论文
2.检索结果中的第一篇论文的结论是什么、创新点是什么、该论文有哪些可改进的地方
3.如何理解**这个名词
4.请针对论文研究的问题，帮我制定一个学习计划，并给出学习资料
5.哪些论文提到了QCMP形成、找出与QCMP相关的图
```




