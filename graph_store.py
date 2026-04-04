"""
graph_store.py — DuckDB-backed knowledge graph for Living Memory

Tables:
  nodes           — entities with JSON properties and FLOAT[N] embeddings
  edges           — directed weighted relationships between nodes
  adjacency_cache — precomputed bidirectional multi-hop neighbourhood

The VSS extension provides an HNSW index on node embeddings for fast
semantic_search. All write paths hold self._lock so DuckDB access is
single-threaded. The _executor serialises fire-and-forget writes from the
synchronous voice pipeline without blocking audio playback.
"""

import asyncio
import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DB_PATH       = Path(__file__).parent / "graph.db"
EMBEDDING_DIM = int(os.getenv("LM_EMBEDDING_DIM", "768"))

# Controlled relationship vocabulary — living-memory.md: ~15 types max
ALLOWED_RELATIONSHIPS: frozenset[str] = frozenset({
    "knows", "works_at", "lives_in", "part_of", "related_to",
    "created_by", "used_by", "located_in", "owned_by", "instance_of",
    "caused_by", "followed_by", "contradicts", "supports", "mentions",
})

# ---------------------------------------------------------------------------
# SCHEMA DDL
# ---------------------------------------------------------------------------

_SCHEMA_STMTS = [
    "CREATE SEQUENCE IF NOT EXISTS nodes_seq START 1",
    "CREATE SEQUENCE IF NOT EXISTS edges_seq START 1",
    f"""
    CREATE TABLE IF NOT EXISTS nodes (
        id         BIGINT    DEFAULT nextval('nodes_seq') PRIMARY KEY,
        name       TEXT      NOT NULL,
        label      TEXT      NOT NULL,
        properties JSON,
        embedding  FLOAT[{EMBEDDING_DIM}],
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        id           BIGINT DEFAULT nextval('edges_seq') PRIMARY KEY,
        from_id      BIGINT NOT NULL,
        to_id        BIGINT NOT NULL,
        relationship TEXT   NOT NULL,
        weight       REAL   DEFAULT 1.0,
        created_at   TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique ON edges(from_id, to_id, relationship)",
    """
    CREATE TABLE IF NOT EXISTS adjacency_cache (
        node_id     BIGINT  NOT NULL,
        neighbor_id BIGINT  NOT NULL,
        hops        INTEGER NOT NULL,
        PRIMARY KEY (node_id, neighbor_id)
    )
    """,
]

_HNSW_STMT = "CREATE INDEX IF NOT EXISTS nodes_embedding_hnsw ON nodes USING HNSW(embedding)"

# ---------------------------------------------------------------------------
# ADJACENCY CACHE REBUILD
# Bidirectional, up to 2 hops, stores min hop count for each node pair.
# Covers five traversal patterns:
#   1-hop forward / backward
#   2-hop forward→forward  (A→B→C)
#   2-hop backward→forward (A←B→C, siblings)
#   2-hop forward→backward (A→B←C, co-neighbours)
# ---------------------------------------------------------------------------

_REFRESH_STMTS = [
    "DELETE FROM adjacency_cache",
    """
    INSERT INTO adjacency_cache (node_id, neighbor_id, hops)
    SELECT node_id, neighbor_id, min(hops) FROM (
        SELECT from_id AS node_id, to_id   AS neighbor_id, 1 AS hops FROM edges
        UNION ALL
        SELECT to_id   AS node_id, from_id AS neighbor_id, 1 AS hops FROM edges
        UNION ALL
        SELECT e1.from_id AS node_id, e2.to_id   AS neighbor_id, 2 AS hops
          FROM edges e1 JOIN edges e2 ON e1.to_id  = e2.from_id
        UNION ALL
        SELECT e1.to_id   AS node_id, e2.to_id   AS neighbor_id, 2 AS hops
          FROM edges e1 JOIN edges e2 ON e1.from_id = e2.from_id
        UNION ALL
        SELECT e1.from_id AS node_id, e2.from_id AS neighbor_id, 2 AS hops
          FROM edges e1 JOIN edges e2 ON e1.to_id   = e2.to_id
    ) sub
    GROUP BY node_id, neighbor_id
    HAVING node_id != neighbor_id
    """,
]

# ---------------------------------------------------------------------------
# GRAPH STORE
# ---------------------------------------------------------------------------


class GraphStore:
    """
    Thread-safe DuckDB knowledge graph.

    Synchronous callers (voice_pipeline.py) use submit_* for fire-and-forget
    writes. Async callers use async_* coroutines. Direct calls block.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._conn = duckdb.connect(str(db_path))
        self._lock = threading.Lock()
        # Single worker keeps writes serialised; caller is never blocked.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="graph_writer"
        )
        self._init_schema()

    # ------------------------------------------------------------------
    # SCHEMA INITIALISATION
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            try:
                self._conn.execute("INSTALL vss")
            except duckdb.IOException:
                pass  # already installed — safe to continue offline
            self._conn.execute("LOAD vss")
            for stmt in _SCHEMA_STMTS:
                self._conn.execute(stmt)
            # HNSW persistence is required for file-based databases
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")
            # HNSW index must be created after VSS is loaded
            self._conn.execute(_HNSW_STMT)
            log.debug("Schema initialised")

    # ------------------------------------------------------------------
    # NODES
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        name: str,
        label: str,
        properties: dict | None = None,
    ) -> int:
        """
        Insert a new node or update label/properties if name already exists.
        Name is stripped and used as the deduplication key.
        Returns the node id.
        """
        key = name.strip()
        props_json = json.dumps(properties or {})
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO nodes (name, label, properties)
                VALUES (?, ?, ?::JSON)
                ON CONFLICT (name) DO UPDATE SET
                    label      = EXCLUDED.label,
                    properties = EXCLUDED.properties,
                    updated_at = now()
                """,
                [key, label.strip(), props_json],
            )
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE name = ?", [key]
            ).fetchone()
            return row[0]

    def update_node_embedding(self, name: str, embedding: list[float]) -> None:
        """Store a pre-computed embedding vector on an existing node."""
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dimensional, got {len(embedding)}"
            )
        with self._lock:
            self._conn.execute(
                f"UPDATE nodes SET embedding = ?::FLOAT[{EMBEDDING_DIM}] WHERE name = ?",
                [embedding, name.strip()],
            )

    # ------------------------------------------------------------------
    # EDGES
    # ------------------------------------------------------------------

    def add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        """
        Add or update a directed weighted edge.
        Endpoint nodes are auto-created with label 'Unknown' if absent.
        Rebuilds the adjacency cache after every successful write.
        """
        rel = relationship.lower().strip()
        if rel not in ALLOWED_RELATIONSHIPS:
            raise ValueError(
                f"Relationship '{rel}' not in controlled vocabulary. "
                f"Allowed: {sorted(ALLOWED_RELATIONSHIPS)}"
            )
        with self._lock:
            for nm in (from_name.strip(), to_name.strip()):
                self._conn.execute(
                    "INSERT INTO nodes (name, label) VALUES (?, 'Unknown') ON CONFLICT (name) DO NOTHING",
                    [nm],
                )
            from_id = self._conn.execute(
                "SELECT id FROM nodes WHERE name = ?", [from_name.strip()]
            ).fetchone()[0]
            to_id = self._conn.execute(
                "SELECT id FROM nodes WHERE name = ?", [to_name.strip()]
            ).fetchone()[0]
            self._conn.execute(
                """
                INSERT INTO edges (from_id, to_id, relationship, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (from_id, to_id, relationship)
                DO UPDATE SET weight = EXCLUDED.weight
                """,
                [from_id, to_id, rel, float(weight)],
            )
            self._refresh_adjacency_cache()

    def _refresh_adjacency_cache(self) -> None:
        """Rebuild full adjacency cache. Must be called while holding self._lock."""
        for stmt in _REFRESH_STMTS:
            self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    def query_neighbors(self, node_name: str, hops: int = 2) -> list[dict]:
        """
        Return all nodes reachable from node_name within N hops.
        Traversal is bidirectional (both in-edges and out-edges followed).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE name = ?", [node_name.strip()]
            ).fetchone()
            if not row:
                return []
            node_id = row[0]
            rows = self._conn.execute(
                """
                SELECT n.name, n.label, n.properties, ac.hops
                  FROM adjacency_cache ac
                  JOIN nodes n ON ac.neighbor_id = n.id
                 WHERE ac.node_id = ? AND ac.hops <= ?
                 ORDER BY ac.hops, n.name
                """,
                [node_id, hops],
            ).fetchall()
        return [
            {
                "name":       r[0],
                "label":      r[1],
                "properties": json.loads(r[2]) if r[2] else {},
                "hops":       r[3],
            }
            for r in rows
        ]

    def semantic_search(
        self, query_embedding: list[float], top_n: int = 10
    ) -> list[dict]:
        """
        Return the top_n nodes most similar to query_embedding by cosine
        similarity. Uses the HNSW index; nodes without an embedding are
        excluded. Caller must supply a unit-normalised vector for best results.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding must be {EMBEDDING_DIM}-dimensional, "
                f"got {len(query_embedding)}"
            )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT name, label, properties,
                       array_cosine_similarity(
                           embedding, ?::FLOAT[{EMBEDDING_DIM}]
                       ) AS score
                  FROM nodes
                 WHERE embedding IS NOT NULL
                 ORDER BY score DESC
                 LIMIT ?
                """,
                [query_embedding, top_n],
            ).fetchall()
        return [
            {
                "name":       r[0],
                "label":      r[1],
                "properties": json.loads(r[2]) if r[2] else {},
                "score":      float(r[3]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # ASYNC / FIRE-AND-FORGET WRAPPERS
    # ------------------------------------------------------------------

    async def async_upsert_node(
        self, name: str, label: str, properties: dict | None = None
    ) -> int:
        return await asyncio.to_thread(self.upsert_node, name, label, properties)

    async def async_add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        await asyncio.to_thread(self.add_edge, from_name, to_name, relationship, weight)

    def submit_upsert_node(
        self, name: str, label: str, properties: dict | None = None
    ) -> Future:
        """
        Non-blocking upsert for use in voice_pipeline.py.
        Returns a Future; callers can ignore it or call .result() to wait.
        """
        return self._executor.submit(self.upsert_node, name, label, properties)

    def submit_add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> Future:
        return self._executor.submit(
            self.add_edge, from_name, to_name, relationship, weight
        )

    # ------------------------------------------------------------------
    # UTILITY QUERIES
    # ------------------------------------------------------------------

    def stats(self) -> tuple[int, int]:
        """Return (node_count, edge_count) for the live graph growth counter."""
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            e = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return int(n), int(e)

    def search_nodes_by_name(self, fragment: str, limit: int = 5) -> list[dict]:
        """Case-insensitive substring search over node names."""
        pattern = f"%{fragment.strip()}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, label, properties FROM nodes WHERE name ILIKE ? LIMIT ?",
                [pattern, limit],
            ).fetchall()
        return [
            {
                "name":       r[0],
                "label":      r[1],
                "properties": json.loads(r[2]) if r[2] else {},
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# SMOKE TEST  (venv/bin/python graph_store.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_db = Path(__file__).parent / "graph_test.db"
    test_db.unlink(missing_ok=True)

    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)
            log.error("FAIL: %s", msg)
        else:
            log.info("OK:   %s", msg)

    with GraphStore(db_path=test_db) as gs:

        # ── nodes & deduplication ──────────────────────────────────────
        id_marie  = gs.upsert_node("Marie Curie", "Person", {"occupation": "physicist"})
        id_paris  = gs.upsert_node("Paris",       "Place",  {"country": "France"})
        id_radium = gs.upsert_node("Radium",      "Concept", {"type": "element"})

        id_marie2 = gs.upsert_node("Marie Curie", "Person", {"occupation": "chemist"})
        check(id_marie == id_marie2,          "dedup: second upsert preserves id")

        row = gs._conn.execute(
            "SELECT properties FROM nodes WHERE name = 'Marie Curie'"
        ).fetchone()
        props = json.loads(row[0])
        check(props.get("occupation") == "chemist", "dedup: properties updated on conflict")

        # ── edges & adjacency cache ────────────────────────────────────
        gs.add_edge("Marie Curie", "Paris",  "lives_in",  weight=0.9)
        gs.add_edge("Marie Curie", "Radium", "created_by",weight=1.0)
        gs.add_edge("Radium",      "Paris",  "related_to",weight=0.5)

        n1 = gs.query_neighbors("Marie Curie", hops=1)
        check(len(n1) == 2,  f"1-hop neighbours of Marie Curie: expected 2, got {len(n1)}")

        n2 = gs.query_neighbors("Marie Curie", hops=2)
        check(len(n2) >= 2,  f"2-hop neighbours of Marie Curie: expected ≥2, got {len(n2)}")

        n_paris = gs.query_neighbors("Paris", hops=1)
        check(len(n_paris) >= 2, f"Paris 1-hop (bidirectional): expected ≥2, got {len(n_paris)}")

        # ── embeddings & semantic search ───────────────────────────────
        rng = random.Random(42)

        def rand_unit_vec() -> list[float]:
            v = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
            mag = sum(x ** 2 for x in v) ** 0.5
            return [x / mag for x in v]

        for nm in ("Marie Curie", "Paris", "Radium"):
            gs.update_node_embedding(nm, rand_unit_vec())

        results = gs.semantic_search(rand_unit_vec(), top_n=3)
        check(len(results) == 3,            "semantic_search returns 3 results")
        check(all("score" in r for r in results), "semantic_search results have scores")
        check(results[0]["score"] >= results[-1]["score"], "results ordered by score desc")

        # ── HNSW index exists in catalogue ────────────────────────────
        idx_row = gs._conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'nodes_embedding_hnsw'"
        ).fetchone()
        check(idx_row is not None, "HNSW index exists in duckdb_indexes()")

        # ── fire-and-forget ───────────────────────────────────────────
        f = gs.submit_upsert_node("London", "Place", {})
        f.result()  # wait in test context
        row = gs._conn.execute("SELECT id FROM nodes WHERE name = 'London'").fetchone()
        check(row is not None, "fire-and-forget upsert: London inserted")

        # ── bad relationship is rejected ──────────────────────────────
        try:
            gs.add_edge("Marie Curie", "London", "invented", weight=1.0)
            check(False, "ValueError raised for unknown relationship")
        except ValueError:
            check(True, "ValueError raised for unknown relationship")

    test_db.unlink(missing_ok=True)

    if errors:
        log.error("%d check(s) failed.", len(errors))
        sys.exit(1)

    print("All checks passed.")
    sys.exit(0)
