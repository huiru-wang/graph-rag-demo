from flask import Flask, render_template, request, jsonify, Response
import logging
import sys
from llm_engine import LLMEngine
from memory_store import MemoryStore
from graph_store import GraphStore
from dotenv import load_dotenv
import json
import time
from io import BytesIO

# 设置日志配置，只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
# 延迟初始化避免 Kuzu 锁问题
memory = None
graph = None


def get_stores():
    global memory, graph
    if memory is None:
        memory = MemoryStore()
    if graph is None:
        graph = GraphStore()
    return memory, graph


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_stream", methods=["POST"])
def process_stream():
    data = request.json
    text = data.get("text")
    user_id = data.get("user_id", "default_user")

    def event_stream():
        memory_store, graph_store = get_stores()

        # 发送开始日志
        yield f"data: {json.dumps({'type': 'log', 'message': '开始处理用户输入...', 'timestamp': time.time()})}\n\n"

        # 1. 提取
        yield f"data: {json.dumps({'type': 'log', 'message': '1. ========== 数据提取 ==========', 'timestamp': time.time()})}\n\n"
        new_facts = LLMEngine.extract_facts(text, user_id)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[用户输入]: {text}', 'timestamp': time.time()})}\n\n"
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 事实提取: {new_facts}', 'timestamp': time.time()})}\n\n"

        new_rels = LLMEngine.extract_entities(text, user_id)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 实体关系提取: {new_rels}', 'timestamp': time.time()})}\n\n"

        # 2. 数据召回 (带上 user_id)
        yield f"data: {json.dumps({'type': 'log', 'message': '2. ========== 数据召回 ==========', 'timestamp': time.time()})}\n\n"
        old_memories = memory_store.search_facts(user_id, text)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 记忆召回: {old_memories}', 'timestamp': time.time()})}\n\n"

        entities = list(
            set([r["subject"] for r in new_rels] + [r["object"] for r in new_rels])
        )

        old_rels = graph_store.get_related_rels(user_id, entities)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 实体关系召回: {old_rels}', 'timestamp': time.time()})}\n\n"

        # 3. 冲突检测
        yield f"data: {json.dumps({'type': 'log', 'message': '3. ========== 冲突检测 ==========', 'timestamp': time.time()})}\n\n"
        mem_decisions = LLMEngine.decide_memory_updates(
            new_facts, old_memories, user_id
        )
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 记忆决策: {mem_decisions}', 'timestamp': time.time()})}\n\n"

        graph_decisions = LLMEngine.decide_graph_updates(
            text, new_rels, old_rels, old_memories, user_id
        )
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 实体关系决策: {graph_decisions}', 'timestamp': time.time()})}\n\n"

        # 4. 执行原子化更新 (带上 user_id)
        yield f"data: {json.dumps({'type': 'log', 'message': '4. ========== 原子更新 ==========', 'timestamp': time.time()})}\n\n"
        memory_store.update(user_id, mem_decisions)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 记忆更新完成', 'timestamp': time.time()})}\n\n"

        graph_store.execute_actions(user_id, graph_decisions)
        yield f"data: {json.dumps({'type': 'log', 'message': f'[{user_id}] 实体关系更新完成', 'timestamp': time.time()})}\n\n"

        # 发送完成信号
        yield f"data: {json.dumps({'type': 'complete', 'message': '处理完成', 'timestamp': time.time()})}\n\n"

    return Response(event_stream(), mimetype="text/plain")


@app.route("/favicon.ico")
def favicon():
    # 返回一个大脑emoji作为favicon
    # 使用SVG格式并转换为数据URL
    svg_favicon = """<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
        <rect width="16" height="16" fill="white"/>
        <text x="8" y="14" font-size="14" text-anchor="middle" fill="black">🧠</text>
    </svg>"""

    return Response(svg_favicon, mimetype="image/svg+xml")


@app.route("/chat_with_memory", methods=["POST"])
def chat_with_memory():
    data = request.json
    user_message = data.get("message")
    user_id = data.get("user_id", "default_user")

    def event_stream():
        memory_store, graph_store = get_stores()

        # 1. 检索阶段
        yield f"data: {json.dumps({'type': 'content', 'content': '🔍 正在检索您的记忆数据...\n', 'timestamp': time.time()})}\n\n"

        # 2. 格式化阶段
        memory = search_memory(user_id)

        # 3. 注入阶段 - 组装系统提示
        yield f"data: {json.dumps({'type': 'content', 'content': '🤖 正在向大模型发送请求...\n', 'timestamp': time.time()})}\n\n"

        # 使用LLMEngine中的流式调用方法
        generator = LLMEngine.chat_with_memory(
            memory["formatted_facts"],
            memory["formatted_relations"],
            user_message,
            user_id,
        )
        for chunk in generator():
            yield chunk

    return Response(event_stream(), mimetype="text/plain")


def search_memory(user_id):
    """
    搜索并返回用户的记忆和关系
    """
    memory_store, graph_store = get_stores()

    # 从MemoryStore获取用户事实
    facts = memory_store.get_memories_by_user_id(user_id)

    # 提取事实文本
    facts_list = [fact["text"] for fact in facts]

    # 获取该用户的所有关系
    relations = graph_store.get_all_for_viz(user_id)

    # 提取关系三元组
    relations_list = []
    for item in relations:
        if (
            "data" in item
            and "source" in item["data"]
            and "target" in item["data"]
            and "label" in item["data"]
        ):
            relations_list.append(
                {
                    "subject": item["data"]["source"],
                    "relation": item["data"]["label"],
                    "object": item["data"]["target"],
                }
            )

    # 格式化facts和relations
    formatted_facts = (
        "\\n".join(f"- {fact}" for fact in facts_list)
        if facts_list
        else "No facts stored"
    )
    formatted_relations = (
        "\\n".join(
            f"- ({rel['subject']}, {rel['relation']}, {rel['object']})"
            for rel in relations_list
        )
        if relations_list
        else "No relations stored"
    )

    return {
        "formatted_facts": formatted_facts,
        "formatted_relations": formatted_relations,
    }


@app.route("/graph_data")
def graph_data():
    user_id = request.args.get("user_id", "default_user")

    _, gs = get_stores()
    graph_data = gs.get_all_for_viz(user_id)
    logger.info(f"[{user_id}] 召回 {len(graph_data)} 条实体关系")

    return jsonify(graph_data)


@app.route("/memories")
def get_memories():
    user_id = request.args.get("user_id", "default_user")
    limit = request.args.get("limit", type=int)  # 可选的限制参数

    logger.info(f"Retrieving memories for user_id: {user_id}, limit: {limit}")

    memory_store, _ = get_stores()
    memories = memory_store.get_memories_by_user_id(user_id, limit)

    logger.info(f"[{user_id}] 召回 {len(memories)} 条记忆")

    return jsonify(memories)


if __name__ == "__main__":
    # 关键：debug=True 时必须禁用 reloader，否则 Kuzu 会报锁错误
    logger.info("Starting Flask application on port 5003...")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8000)
