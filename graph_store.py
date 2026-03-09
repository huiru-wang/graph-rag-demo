import kuzu
import logging
import sys
import os


# 设置日志配置，只输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class GraphStore:
    def __init__(self, db_path=".data/kuzu_db"):
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        logger = logging.getLogger(__name__)
        try:
            self.conn.execute(
                "CREATE NODE TABLE Entity(name STRING, PRIMARY KEY (name))"
            )
            # 增加 user_id 属性到关系表
            self.conn.execute(
                "CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, type STRING, user_id STRING)"
            )
        except Exception as e:
            logger.warning(f"Schema creation failed (may already exist): {e}")

    def get_related_rels(self, user_id, entities):
        logger = logging.getLogger(__name__)
        rels = []
        for entity in entities:
            # 增加 user_id 过滤
            logger.debug(f"Querying relations for entity: {entity}")
            res = self.conn.execute(
                """
                MATCH (a:Entity {name: $n})-[r:RELATED_TO {user_id: $uid}]->(b:Entity) 
                RETURN a.name, r.type, b.name
                """,
                {"n": entity, "uid": user_id},
            )
            while res.has_next():
                row = res.get_next()
                rels.append({"subject": row[0], "relation": row[1], "object": row[2]})
                logger.debug(f"Found relation: {row[0]} - {row[1]} - {row[2]}")
        return rels

    def execute_actions(self, user_id, actions):
        for act in actions:
            s, r, o = act["subject"], act["relation"], act["object"]
            if act["action"] == "ADD":
                self.conn.execute("MERGE (a:Entity {name: $s})", {"s": s})
                self.conn.execute("MERGE (b:Entity {name: $o})", {"o": o})
                # 创建关系时带上 user_id
                self.conn.execute(
                    """
                    MATCH (a:Entity {name: $s}), (b:Entity {name: $o}) 
                    MERGE (a)-[:RELATED_TO {type: $r, user_id: $uid}]->(b)
                    """,
                    {"s": s, "o": o, "r": r, "uid": user_id},
                )
            elif act["action"] == "DELETE":
                self.conn.execute(
                    """
                    MATCH (a:Entity {name: $s})-[r:RELATED_TO {type: $r, user_id: $uid}]->(b:Entity {name: $o}) 
                    DELETE r
                    """,
                    {"s": s, "o": o, "r": r, "uid": user_id},
                )

    def get_all_for_viz(self, user_id):
        # 仅获取属于该用户的边
        edges_res = self.conn.execute(
            "MATCH (a:Entity)-[r:RELATED_TO {user_id: $uid}]->(b:Entity) RETURN a.name, b.name, r.type",
            {"uid": user_id},
        )
        edges = []
        nodes_set = set()
        while edges_res.has_next():
            row = edges_res.get_next()
            edges.append(
                {"data": {"source": row[0], "target": row[1], "label": row[2]}}
            )
            nodes_set.add(row[0])
            nodes_set.add(row[1])

        nodes = [{"data": {"id": n, "label": n}} for n in nodes_set]
        return nodes + edges
