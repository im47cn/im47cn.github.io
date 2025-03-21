---
title: 如何使用Qdrant和MCP实现LLM的长期记忆功能
author: dreambt
date: 2025-03-21 08:17:00 +0800
categories: [LLM]
tags: [LLM, 数据库, Qdrant, MCP, 向量数据库]
description: 本文全面介绍如何利用Qdrant向量数据库与模型上下文协议(MCP)构建高效的LLM长期记忆系统，包含架构设计、实现步骤和应用场景详解。
---

## 1. 引言：LLM的记忆挑战与解决方案

大型语言模型(LLM)在自然语言理解和生成方面展现出惊人的能力，但它们仍然面临一个核心挑战：缺乏有效的**长期记忆**机制。在对话过程中，LLM通常只能依赖于当前对话上下文中的信息，一旦超出上下文窗口限制，之前的交互信息就会丢失，这严重限制了LLM在需要持续上下文的应用场景中的表现。

**想象一下这个场景**：用户与AI助手进行了一次深入的技术讨论，涉及多个复杂概念。一周后，用户希望继续之前的讨论，但AI助手已经"忘记"了之前的对话内容，需要用户重新解释背景信息。这种体验显然不够理想，尤其在知识管理、客户服务等领域，记忆能力的缺失会大大降低AI系统的实用性。

本文将探讨如何通过结合**Qdrant向量数据库**与**模型上下文协议(MCP)**来构建一个高效的LLM长期记忆系统。这种方案不仅解决了记忆持久化问题，还提供了灵活的检索机制，使AI系统能够智能地回忆与当前对话相关的历史信息。

## 2. 技术背景：核心组件解析

### 2.1 Qdrant向量数据库简介

[Qdrant](https://qdrant.tech/)是一个为向量相似性搜索和记忆管理而优化的开源向量数据库。其核心特性包括：

- **高性能向量搜索**：支持多种相似度度量方法（余弦相似度、欧几里得距离等）
- **强大的过滤能力**：支持结构化数据与向量搜索的结合
- **水平扩展**：支持分布式部署，可处理大规模数据
- **实时更新**：支持动态添加、修改和删除数据
- **跨平台**：提供Docker容器和多平台二进制文件

Qdrant的设计理念是将非结构化数据（如文本、图像）转换为向量表示，然后通过向量相似性来实现高效检索，这与LLM的记忆需求高度契合。

### 2.2 MCP简介

模型上下文协议(Model Context Protocol, MCP)是一种标准化协议，用于AI模型与外部工具和数据源的交互。MCP的核心价值在于：

- **统一性**：提供标准接口，使AI模型能够以一致的方式连接各种数据源和工具
- **灵活性**：支持多种数据类型和工具操作，适应不同应用场景
- **生态系统**：丰富的插件生态，可以扩展AI模型的功能范围
- **数据安全**：敏感数据可保留在本地处理，增强数据隐私保护

MCP架构包含三个关键组件：Host（如Claude Desktop）、Client（与AI模型交互）和Server（提供具体功能实现）。

### 2.3 向量数据库与LLM记忆的关系

向量数据库在LLM记忆系统中扮演着核心角色，其工作原理可概括为：

1. **文本嵌入**：将文本信息转换为高维向量表示（嵌入）
2. **相似性搜索**：基于向量相似度快速检索相关信息
3. **上下文重建**：从检索结果重建LLM所需的上下文信息

这一过程使LLM能够"回忆"之前存储的信息，而不受上下文窗口限制，从而实现长期记忆能力。

## 3. 原理详解：理解记忆系统的工作机制

### 3.1 向量嵌入与语义搜索基础

向量嵌入技术是LLM记忆系统的基础，它将自然语言转换为计算机可理解的数值表示。在实现过程中，我们通常使用专门的嵌入模型（如OpenAI的text-embedding-ada-002或开源的BERT/Sentence-BERT）来生成向量。

以下是向量嵌入的基本流程：

1. **文本预处理**：清洗、分割文本
2. **向量生成**：通过嵌入模型将文本转换为向量（通常是数百或数千维的浮点数数组）
3. **向量存储**：将向量及相关元数据存入向量数据库

语义搜索则是利用这些向量表示进行相似度比较，从而找到语义相关的内容。与传统的基于关键词的搜索不同，语义搜索能够理解内容的含义，即使使用不同的词汇表达相似的概念也能被正确识别。

### 3.2 LLM记忆系统的架构设计

一个完整的LLM记忆系统通常包含以下核心组件：

1. **对话管理器**：处理用户输入和LLM输出，管理对话流程
2. **记忆存储器**：基于向量数据库实现的持久化存储层
3. **记忆检索器**：负责从存储层检索相关记忆
4. **上下文构建器**：基于当前对话和检索到的记忆构建LLM的输入上下文
5. **策略控制器**：决定何时存储新记忆、检索哪些记忆等

这种架构设计使LLM能够智能地管理和利用历史信息，提供连贯、个性化的用户体验。

### 3.3 Qdrant与MCP的集成机制

MCP框架可以很好地与Qdrant集成，形成一个强大的LLM记忆服务。其工作流程如下：

1. **MCP Server**创建到Qdrant的连接，并提供一系列工具（Tools）：
   - 存储记忆工具（将对话内容嵌入并存储）
   - 检索记忆工具（检索相关历史信息）
   - 记忆管理工具（更新、删除记忆等）

2. **LLM**（如Claude）通过MCP Client调用这些工具，实现：
   - 在对话过程中自动存储重要信息
   - 在需要时检索相关历史记忆
   - 根据检索到的记忆生成更具上下文感知的回复

3. **用户**通过Host（如Claude Desktop）与系统交互，获得连贯一致的对话体验

这种集成方式的优势在于，它遵循标准化协议，可以轻松地适配不同的LLM和前端应用，同时保持高性能和灵活性。

## 4. 实现步骤：从理论到实践

### 4.1 环境准备

在开始实现之前，需要准备以下环境：

```bash
# 安装必要的Python包
pip install qdrant-client sentence-transformers fastapi uvicorn mcp

# 安装Qdrant（使用Docker）
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

此外，确保已安装支持MCP的客户端（如Claude Desktop）和Python 3.10+环境。

### 4.2 Qdrant配置

首先，我们需要配置Qdrant来存储对话记忆。以下是基本配置步骤：

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 连接到Qdrant服务器
client = QdrantClient("localhost", port=6333)

# 创建记忆集合
client.recreate_collection(
    collection_name="conversation_memories",
    vectors_config=models.VectorParams(
        size=1536,  # OpenAI Ada embedding的维度
        distance=models.Distance.COSINE  # 使用余弦相似度
    )
)
```

这段代码创建了一个名为"conversation_memories"的集合，用于存储对话记忆的向量表示。

### 4.3 构建MCP Server

接下来，我们实现一个基于MCP的记忆服务。以下是核心实现：

```python
import os
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 初始化MCP服务器
mcp = FastMCP("LLM记忆服务")

# 初始化向量嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 连接Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

@mcp.tool()
def store_memory(
    user_id: str,
    content: str,
    metadata: Dict[str, Any] = None
) -> str:
    """
    存储用户对话内容作为记忆
    
    Args:
        user_id: 用户ID
        content: 对话内容
        metadata: 额外元数据（时间戳、话题等）
    
    Returns:
        记忆ID
    """
    # 生成向量嵌入
    vector = model.encode(content).tolist()
    
    # 准备元数据
    payload = metadata or {}
    payload.update({
        "user_id": user_id,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    
    # 存储到Qdrant
    result = qdrant_client.upsert(
        collection_name="conversation_memories",
        points=[
            {
                "id": str(np.random.randint(0, 10000000)),
                "vector": vector,
                "payload": payload
            }
        ]
    )
    
    return f"成功存储记忆: {result}"

@mcp.tool()
def retrieve_memories(
    user_id: str,
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    检索用户相关记忆
    
    Args:
        user_id: 用户ID
        query: 查询内容
        limit: 返回结果数量上限
    
    Returns:
        相关记忆列表
    """
    # 生成查询向量
    query_vector = model.encode(query).tolist()
    
    # 从Qdrant检索相似向量
    search_result = qdrant_client.search(
        collection_name="conversation_memories",
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        ),
        limit=limit
    )
    
    # 转换结果格式
    memories = []
    for result in search_result:
        memory = result.payload
        memory["score"] = result.score
        memories.append(memory)
    
    return memories

@mcp.tool()
def clear_user_memories(user_id: str) -> str:
    """
    清除指定用户的所有记忆
    
    Args:
        user_id: 用户ID
    
    Returns:
        操作结果描述
    """
    result = qdrant_client.delete(
        collection_name="conversation_memories",
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )
    )
    
    return f"已清除用户{user_id}的所有记忆"

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
```

这段代码实现了三个核心工具：
- `store_memory`: 存储对话内容作为记忆
- `retrieve_memories`: 检索与当前对话相关的历史记忆
- `clear_user_memories`: 清除用户的记忆（用于隐私保护或测试）

### 4.4 配置MCP与Claude集成

要将我们的MCP Server与Claude Desktop集成，需要修改Claude的配置文件：

```bash
# 编辑Claude Desktop配置（MacOS）
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

添加以下配置：

```json
{
  "mcpServers": {
    "llm_memory": {
      "command": "/path/to/python",
      "args": [
        "/path/to/your/memory_server.py"
      ]
    }
  }
}
```

重启Claude Desktop后，我们的记忆服务将可用于Claude模型。

## 5. 完整示例：构建一个具有记忆的AI助手

下面我们将展示一个完整的使用案例，演示如何结合MCP和Qdrant构建一个具有长期记忆能力的AI助手。

### 5.1 记忆服务实现

首先，我们创建一个更完整的记忆服务实现，增加会话管理和上下文构建功能：

```python
# memory_server.py

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# 初始化MCP服务器
mcp = FastMCP("AI记忆助手")

# 初始化向量嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 连接Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

# 确保集合存在
try:
    qdrant_client.get_collection("conversation_memories")
except Exception:
    qdrant_client.recreate_collection(
        collection_name="conversation_memories",
        vectors_config=models.VectorParams(
            size=384,  # all-MiniLM-L6-v2的维度
            distance=models.Distance.COSINE
        )
    )

# 会话管理
sessions = {}

@mcp.tool()
def start_session(user_id: str) -> str:
    """
    开始一个新的对话会话
    
    Args:
        user_id: 用户ID
    
    Returns:
        会话ID
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_id": user_id,
        "start_time": datetime.now().isoformat(),
        "messages": []
    }
    return session_id

@mcp.tool()
def add_message(
    session_id: str,
    role: str,
    content: str,
    store_as_memory: bool = True
) -> str:
    """
    添加消息到会话并可选地存储为记忆
    
    Args:
        session_id: 会话ID
        role: 消息发送者角色 (user/assistant)
        content: 消息内容
        store_as_memory: 是否存储为长期记忆
    
    Returns:
        操作结果
    """
    if session_id not in sessions:
        return f"会话{session_id}不存在"
    
    # 添加到会话
    timestamp = datetime.now().isoformat()
    sessions[session_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })
    
    # 存储为长期记忆
    if store_as_memory:
        user_id = sessions[session_id]["user_id"]
        
        # 生成向量嵌入
        vector = model.encode(content).tolist()
        
        # 准备元数据
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
        
        # 存储到Qdrant
        point_id = str(uuid.uuid4())
        qdrant_client.upsert(
            collection_name="conversation_memories",
            points=[
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        )
    
    return "消息已添加" + (" 并存储为记忆" if store_as_memory else "")

@mcp.tool()
def retrieve_relevant_memories(
    user_id: str,
    query: str,
    limit: int = 5,
    time_window_days: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    检索与查询相关的用户记忆
    
    Args:
        user_id: 用户ID
        query: 查询内容
        limit: 返回结果数量上限
        time_window_days: 可选的时间窗口限制（天）
    
    Returns:
        相关记忆列表
    """
    # 生成查询向量
    query_vector = model.encode(query).tolist()
    
    # 构建查询过滤器
    must_conditions = [
        models.FieldCondition(
            key="user_id",
            match=models.MatchValue(value=user_id)
        )
    ]
    
    # 添加时间窗口过滤
    if time_window_days is not None:
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=time_window_days)).isoformat()
        must_conditions.append(
            models.FieldCondition(
                key="timestamp",
                range=models.Range(
                    gte=cutoff_date
                )
            )
        )
    
    # 从Qdrant检索相似向量
    search_result = qdrant_client.search(
        collection_name="conversation_memories",
        query_vector=query_vector,
        query_filter=models.Filter(
            must=must_conditions
        ),
        limit=limit
    )
    
    # 转换结果格式
    memories = []
    for result in search_result:
        memory = result.payload
        memory["relevance_score"] = result.score
        memories.append(memory)
    
    return memories

@mcp.tool()
def build_context_with_memories(
    user_id: str,
    current_query: str,
    limit: int = 3
) -> str:
    """
    构建包含相关记忆的上下文
    
    Args:
        user_id: 用户ID
        current_query: 当前查询
        limit: 包含的记忆数量
    
    Returns:
        增强的上下文
    """
    memories = retrieve_relevant_memories(
        user_id=user_id,
        query=current_query,
        limit=limit
    )
    
    if not memories:
        return "没有找到相关的历史记忆。"
    
    context = "根据您之前的对话，我找到了这些相关信息：\n\n"
    
    for i, memory in enumerate(memories, 1):
        timestamp = datetime.fromisoformat(memory["timestamp"])
        formatted_time = timestamp.strftime("%Y年%m月%d日 %H:%M")
        
        context += f"{i}. 在{formatted_time}，"
        if memory["role"] == "user":
            context += f"您说过：\"{memory['content']}\"\n"
        else:
            context += f"我回复说：\"{memory['content']}\"\n"
    
    context += "\n基于这些历史信息，我可以更好地回答您的问题。"
    
    return context

@mcp.tool()
def clear_user_data(user_id: str, confirm: bool = False) -> str:
    """
    清除用户的所有数据（包括会话和记忆）
    
    Args:
        user_id: 用户ID
        confirm: 确认操作
    
    Returns:
        操作结果
    """
    if not confirm:
        return "操作未确认，没有执行任何删除"
    
    # 清除会话
    sessions_to_remove = []
    for session_id, session in sessions.items():
        if session["user_id"] == user_id:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        sessions.pop(session_id)
    
    # 清除Qdrant中的记忆
    qdrant_client.delete(
        collection_name="conversation_memories",
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id)
                )
            ]
        )
    )
    
    return f"已清除用户{user_id}的所有数据"

if __name__ == "__main__":
    # 运行MCP服务器
    mcp.run()
```

### 5.2 使用LLM与记忆助手交互

现在，让我们来看一个具体使用场景。以下是一个用户与记忆增强的Claude的对话示例：

```
用户: 你好，我是李明。我想聊聊人工智能的发展。

Claude: 你好，李明！很高兴和你聊人工智能的发展。
[Claude调用MCP start_session工具，创建新会话]
[Claude调用MCP add_message工具，记录用户消息]

人工智能正经历快速发展，从早期的专家系统到现在的大型语言模型(LLM)，技术进步显著。

目前的热点领域包括:
1. 大型语言模型的持续进化
2. 多模态AI能力的提升
3. AI在特定领域的垂直应用
4. 强化学习与自主系统

你对哪个方向特别感兴趣？我很乐意深入讨论任何一个领域。
[Claude调用MCP add_message工具，记录自己的回复]

用户: 我对多模态AI很感兴趣。特别是计算机视觉和NLP的结合应用。

Claude: [Claude调用MCP add_message工具，记录用户消息]

多模态AI确实是个令人兴奋的领域！计算机视觉与NLP的结合创造了许多强大的应用场景。这种结合主要体现在以下几个方面：

1. **图像描述与内容理解**：AI系统能够"看懂"图像并生成自然语言描述，如Microsoft的CaptionBot或Google的图像描述技术。

2. **视觉问答系统(VQA)**：允许用户针对图像提出自然语言问题，系统能够理解图像内容并回答问题。

3. **多模态搜索引擎**：用户可以通过文本描述搜索图像，或通过图像搜索相关内容。

4. **辅助技术应用**：帮助视障人士理解视觉内容，将视觉信息转化为语音描述。

5. **增强现实(AR)应用**：结合视觉识别和自然语言处理，创造更自然的AR交互体验。

这些技术背后通常采用"编码器-解码器"架构，将不同模态的信息映射到共享语义空间中。

你对这些应用中的哪一个特别感兴趣？或者你是否正在考虑将多模态AI应用到特定领域？
[Claude调用MCP add_message工具，记录自己的回复]

[一周后...]

用户: 你好，我想继续我们之前关于AI的讨论。

Claude: [Claude调用MCP retrieve_relevant_memories工具，检索之前的对话]
[Claude调用MCP build_context_with_memories工具，构建增强上下文]

你好，李明！很高兴继续我们之前的讨论。

根据我们之前的对话，我记得你对多模态AI特别感兴趣，尤其是计算机视觉和NLP的结合应用。我们讨论了图像描述、视觉问答系统、多模态搜索引擎等多种应用场景。

你想深入探讨多模态AI的哪个方面？也许是最新的技术发展，或者特定的应用案例？或者你想转向人工智能的其他领域？
[Claude调用MCP add_message工具，记录用户消息和自己的回复]
```

这个示例展示了如何利用Qdrant和MCP实现LLM的长期记忆功能，使AI助手能够记住用户的身份和之前的交互内容，从而提供更加个性化和连贯的对话体验。

## 6. 应用场景与效果展示

基于Qdrant和MCP构建的LLM记忆系统可以应用于多种场景：

### 6.1 个人AI助手

个人AI助手通过记忆功能可以：
- 记住用户的偏好、习惯和背景信息
- 追踪长期目标和任务进展
- 提供个性化的建议和提醒
- 在多次对话中保持上下文连贯性

**效果示例**：AI助手能够记住用户上次提到的健身计划，并在数周后的对话中主动询问进展，提供针对性的建议。

### 6.2 企业知识管理

在企业环境中，记忆增强的LLM可以：
- 跟踪项目进展和决策历史
- 整合不同会议、文档和讨论中的信息
- 为新团队成员提供上下文背景
- 保存专业知识，减少信息孤岛

**效果示例**：新加入团队的成员可以咨询AI助手了解项目历史，AI能够整合过去几个月的会议记录、决策过程和关键里程碑，帮助新成员快速融入。

### 6.3 客户服务与支持

在客户服务场景中，记忆系统可以：
- 记住客户的历史问题和喜好
- 跟踪复杂问题的解决过程
- 避免客户重复解释问题
- 提供个性化的服务体验

**效果示例**：即使客户与不同的客服代表交流，AI助手也能记住客户之前报告的问题细节，避免客户重复描述问题，大大提高服务效率和客户满意度。

### 6.4 教育与学习助手

在教育领域，记忆功能可以：
- 跟踪学习者的进度和难点
- 提供个性化的学习路径
- 回顾之前学习的内容
- 建立知识连接，促进深度学习

**效果示例**：AI教学助手能够记住学生上次学习的内容和遇到的困难，根据学生的学习曲线调整教学难度，提供恰到好处的复习和挑战。

## 7. 优化与扩展：提高系统效率和效果

### 7.1 性能优化策略

提高记忆系统性能的关键策略包括：

1. **分层记忆架构**
   - 短期记忆：保留最近对话的完整内容
   - 中期记忆：保存近期重要信息的摘要
   - 长期记忆：存储关键知识点和用户特征

2. **智能记忆管理**
   - 设置记忆优先级（重要信息保留更长时间）
   - 记忆合并与压缩（将相似记忆合并为摘要）
   - 定期记忆回顾与强化（重要信息周期性回顾）

3. **检索优化**
   - 混合检索策略（结合关键词和语义检索）
   - 上下文感知检索（考虑当前对话主题）
   - 记忆预取（预测可能需要的记忆并预先加载）

### 7.2 高级功能扩展

系统可以通过以下方式进一步增强：

1. **情感记忆**
   - 跟踪用户情绪状态
   - 记住触发特定情绪反应的话题
   - 调整回复风格匹配用户情绪

2. **主动回忆**
   - AI主动提及相关历史信息
   - 在适当时机回顾之前的对话
   - 建立新旧信息的连接

3. **记忆解释**
   - 透明展示AI使用了哪些记忆
   - 解释为什么某些记忆被认为相关
   - 允许用户校正错误的记忆

4. **多模态记忆**
   - 存储和检索图像、音频等非文本内容
   - 跨模态关联（文本描述与图像的关联）
   - 多模态上下文理解

### 7.3 隐私与安全考量

实现记忆系统时必须特别注意：

1. **数据隐私保护**
   - 明确用户数据使用政策
   - 提供记忆查看与删除功能
   - 支持定期数据过期机制

2. **安全存储**
   - 敏感信息加密存储
   - 访问控制和身份验证
   - 记忆隔离（不同用户间的记忆严格分离）

3. **伦理使用**
   - 避免记忆偏见累积
   - 定期审计记忆内容
   - 建立伦理使用指南

## 8. 局限性与未来展望

### 8.1 当前局限性

尽管强大，基于向量数据库的记忆系统仍存在一些局限：

1. **上下文理解有限**：向量相似度搜索可能忽略复杂的语境关系
2. **扩展性挑战**：随着记忆量增加，检索效率和质量可能下降
3. **记忆退化**：向量空间中的"概念漂移"可能导致旧记忆检索质量下降
4. **缺乏推理能力**：系统难以进行跨记忆的复杂推理

### 8.2 未来发展方向

记忆系统的未来发展可能包括：

1. **神经符号记忆**：结合向量表示和符号推理
2. **自适应记忆结构**：动态调整记忆组织以适应不同任务
3. **元记忆功能**：AI系统对自身记忆过程的感知和控制
4. **分布式协作记忆**：多个AI系统共享和协作构建记忆网络

## 9. 总结

本文全面探讨了如何结合Qdrant向量数据库与模型上下文协议(MCP)构建高效的LLM长期记忆系统。我们从理论基础出发，详细介绍了向量嵌入与语义搜索原理，设计了完整的系统架构，并提供了实用的实现步骤与代码示例。

通过这种记忆增强方案，LLM能够突破上下文窗口的限制，实现真正的长期记忆能力，为用户提供更加连贯、个性化的交互体验。无论是个人助手、企业知识管理、客户服务还是教育领域，这种记忆能力都能显著提升AI系统的实用性。

虽然当前技术仍有局限，但随着向量数据库和LLM技术的不断发展，我们有理由期待AI系统的记忆能力将越来越接近人类，为各行各业带来更多创新应用。

## 参考资料

1. [Qdrant官方文档](https://qdrant.tech/documentation/)
2. [模型上下文协议(MCP)官方网站](https://modelcontextprotocol.io/)
3. [SentenceTransformers文档](https://www.sbert.net/)
4. [Anthropic Claude API文档](https://docs.anthropic.com/)
5. [向量数据库与LLM应用白皮书](https://arxiv.org/abs/2312.01700)
