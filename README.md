# 个人记忆与知识图谱引擎

这是一个基于大型语言模型（LLM）的个人学习项目，旨在构建一个能够模拟人类记忆和联想能力的智能引擎。它通过持续学习用户输入，动态构建一个个性化的知识图谱和语义记忆库，并能基于这些记忆与用户进行深度对话。

## 核心功能

- **动态知识图谱构建**: 从非结构化的用户输入中实时提取实体和关系，并将其存入 **Kuzu** 图数据库，形成结构化的知识网络。
- **双模记忆存储**:
  - **图谱记忆 (GraphStore)**: 使用 Kuzu 存储精确、结构化的实体关系（例如：“小明，就职于，阿里巴巴”）。
  - **事实记忆 (MemoryStore)**: 使用 **ChromaDB** 向量数据库存储非结构化的事实片段，通过文本向量化实现语义模糊搜索和联想（例如：能够将“我喜欢吃什么”关联到“最爱的食物是披萨”）。
- **基于个性化记忆的智能对话**: 在与用户聊天时，系统会从图谱和向量库中检索相关记忆，并将其注入到 LLM 的上下文中，从而提供高度个性化和情境感知的回答。
- **冲突检测与原子化更新**: 在学习新知识时，引擎会与现有记忆进行比对，通过 LLM 判断是应该新增（ADD）、更新（UPDATE）还是删除（DELETE）信息，避免记忆冲突和冗余。
- **多用户数据隔离**: 所有记忆和图谱数据都通过 `user_id` 进行隔离，确保每个用户拥有独立的、私密的知识库。
- **实时处理反馈**: 采用流式处理（Server-Sent Events），将知识提取、决策和更新的每一步实时反馈给前端，提供透明、可观察的处理过程。

## 技术架构

项目的核心思想是“RAG in, RAG out”，即通过检索增强生成（RAG）的方式来理解和处理输入，再通过同样的方式来生成富有上下文的输出。

### 数据处理流程

```mermaid
graph TD
    A[用户输入文本] --> B{LLM Engine: 提取};
    B --> C[提取事实 (Facts)];
    B --> D[提取实体关系 (Triples)];

    C --> E{MemoryStore (ChromaDB)};
    D --> F{GraphStore (Kuzu)};

    E --> G[向量相似度搜索];
    F --> H[图数据库查询];

    G --> I{LLM Engine: 冲突决策};
    H --> I;
    C --> I;
    D --> I;

    I --> J[生成 ADD/UPDATE/DELETE 决策];
    J --> K[原子化更新 MemoryStore];
    J --> L[原子化更新 GraphStore];

    subgraph "知识提取与存储"
        A
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
    end

    M[用户提问] --> N{Chat with Memory};
    N --> O[检索事实和关系];
    O --> P[注入 System Prompt];
    P --> Q{LLM Engine: 生成回答};
    Q --> R[流式返回答案];

    subgraph "记忆对话"
        M
        N
        O
        P
        Q
        R
    end
```

### 技术栈

- **后端**: Flask
- **图数据库**: Kuzu
- **向量数据库**: ChromaDB
- **LLM 交互**: OpenAI SDK (兼容阿里云通义千问等服务)
- **环境管理**: python-dotenv

## 如何运行

1.  **克隆仓库**
    ```bash
    git clone https://github.com/your-username/graph-rag-demo.git
    cd graph-rag-demo
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    # 或者根据 pyproject.toml 手动安装
    pip install chromadb flask kuzu openai pydantic python-dotenv
    ```

3.  **配置环境**
    在项目根目录下创建一个 `.env` 文件，并填入你的大语言模型服务的 API 密钥和地址。

    ```.env
    cat <<EOF > .env
    OPENAI_API_KEY=sk-549678f8ccee4b86994bbb741d0d9f12
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    EOF
    ```

4.  **启动应用**
    ```bash
    python app.py
    ```
    应用将在 `http://127.0.0.1:8000` 上运行。

    > **注意**: 由于 Kuzu 数据库的锁机制，Flask 应用在 `debug=True` 模式下必须禁用 `reloader` (`use_reloader=False`)，否则会因多进程访问数据库而报错。

## 设计思考：挑战与解决方案

### 1. 事实记忆中的主体丢失问题

在早期版本中，我们发现一个严重问题：当用户连续输入关于不同主体的信息时，记忆系统会发生混淆。

**Bad Case 示例:**

1.  **用户输入**: `我在一家创业公司工作担任服务端开发职位。`
    - **错误记忆**: `担任服务端开发`, `就职于创业公司` (丢失了主体“我”)
2.  **用户输入**: `小明在阿里巴巴工作担任前端开发。`
    - **冲突决策**: 模型可能会错误地认为“担任前端开发”是对“担任服务端开发”的**更新**，从而覆盖掉关于“我”的信息。

**核心原因**: 事实记忆（向量库）没有存储**主体信息**，导致无法区分信息是关于谁的。

**解决方案**:

- **在 Prompt 中强化约束**: 我们在 `FACT_RETRIEVAL_PROMPT` 中明确指示 LLM：
  > "If the fact is about the user, use 'USER' as the subject. If it's about someone else, use their name."
- **引入显式主体**: 要求模型在提取事实时，必须明确事实的主体。例如，将 `“我在...工作”` 提取为 `“USER 在...工作”`。
- **图数据库的天然优势**: 相比之下，图数据库天然要求每个关系都必须有明确的`主-谓-宾`三元组，因此不存在主体丢失的问题。

通过以上改进，模型在进行冲突决策时，会因为主语（`USER` vs `小明`）不一致而正确地选择 `ADD`（新增）而非 `UPDATE`（更新）。

### 2. 记忆检索与注入方案 (RAG)

我们探讨了两种将记忆注入对话的 RAG 方案：

#### 方案一：传统 RAG (已实现)

这是一个直接的“检索 -> 格式化 -> 注入”流程。

1.  **检索 (Retrieval)**: 当用户提问时，系统根据用户的 Query 和 `user_id`，分别从向量库和图数据库中检索相关的记忆。
2.  **格式化 (Formatting)**: 将检索到的结构化数据（JSON）转换为 LLM 能理解的自然语言文本。
3.  **注入 (Injection)**: 将格式化后的文本作为上下文注入到 `System Prompt` 中，引导 LLM 生成回答。

- **优点**: 实现简单、流程清晰。
- **缺点**: 检索效果完全依赖于用户当前问题的字面内容，可能不够智能。

#### 方案二：Agentic RAG (TODO)

这是一种更高级的方案，将检索的决策权交给 LLM。

- **思路**: 不直接为用户检索，而是为 LLM 提供一个或多个“记忆检索工具”（Tools）。当用户提问时，LLM 会自行分析是否需要调用这些工具来获取额外信息。
- **优点**: 模型能够更深刻地理解用户意图，并按需、精准地获取所需记忆，交互更智能、更自然。
- **实现**: 需要基于支持 Function Calling 或 Tool-use 的模型来构建 Agent。

本项目当前实现了方案一，方案二作为未来探索和优化的方向。

## API 端点

- `POST /process_stream`: 接收用户输入文本，流式返回知识提取和处理的全过程。
- `POST /chat_with_memory`: 与用户的个性化记忆进行流式对话。
- `GET /graph_data`: 获取指定用户的知识图谱数据，用于前端可视化。
- `GET /memories`: 获取指定用户的记忆列表。
