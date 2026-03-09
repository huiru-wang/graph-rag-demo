import chromadb
import uuid
from datetime import datetime
from openai import OpenAI
import logging
import sys
import os

# 设置日志配置，只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self):
        os.makedirs(os.path.dirname(os.path.abspath(".data/")), exist_ok=True)
        self.client = chromadb.PersistentClient(path=".data/chroma_db")
        # 如果不指定 embedding_function，Chroma 会使用默认的本地模型
        self.collection = self.client.get_or_create_collection("user_memories")
        self.ai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "sk-d9733c5f1f994def9494b6fbb55232ff"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )

    def _get_embedding(self, text):
        """显式调用模型进行向量化"""
        # 如果使用 OpenAI: text-embedding-3-small
        # 如果使用通义千问: text-embedding-v2
        response = self.ai_client.embeddings.create(
            input=text, model="text-embedding-v2"
        )
        return response.data[0].embedding

    def search_facts(self, user_id, query_text, n=3):
        # 1. 手动将查询文本向量化
        query_vector = self._get_embedding(query_text)

        # 2. 使用 query_embeddings 而不是 query_texts 进行搜索
        results = self.collection.query(
            query_embeddings=[query_vector],  # 传入向量
            n_results=n,
            where={"user_id": user_id},
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                memories.append(
                    {"id": results["ids"][0][i], "text": results["documents"][0][i]}
                )
        return memories

    def update(self, user_id, memory_actions):
        for item in memory_actions:
            event = item.get("event")
            if event == "ADD":
                # 插入时也手动生成向量
                vector = self._get_embedding(item["text"])
                self.collection.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[vector],  # 显式存储向量
                    documents=[item["text"]],
                    metadatas=[
                        {"user_id": user_id, "created_at": datetime.now().isoformat()}
                    ],
                )
            elif event == "UPDATE":
                vector = self._get_embedding(item["text"])
                self.collection.update(
                    ids=[item["id"]], embeddings=[vector], documents=[item["text"]]
                )
            elif event == "DELETE":
                self.collection.delete(ids=[item["id"]])

    def get_memories_by_user_id(self, user_id, limit=None):
        """
        获取指定用户ID的记忆列表，按时间倒序排列
        :param user_id: 用户ID
        :param limit: 返回结果数量限制
        :return: 记忆列表，每项包含id、text和created_at字段
        """
        # 查询指定用户的所有记忆
        results = self.collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],  # 包含文档内容和元数据
        )

        # 组合返回的数据
        memories = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                memory_item = {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "created_at": results["metadatas"][i]["created_at"],
                }
                memories.append(memory_item)

        # 按照创建时间倒序排序
        memories.sort(key=lambda x: x["created_at"], reverse=True)

        # 如果有限制数量，则返回前limit条记录
        if limit:
            memories = memories[:limit]

        return memories
