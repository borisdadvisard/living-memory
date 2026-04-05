"""
graph_store.py — DuckDB-backed knowledge graph for Living Memory

Tables (Universal Graph ontology):
  entities         — people, places, orgs, objects, concepts
  events           — things that happen (deliveries, meetings, etc.)
  commitments      — obligations with owner, due time, status
  state_assertions — time-bounded beliefs about subjects
  sources          — origin of information
  relations        — typed, temporal connections between any two objects
  embeddings       — unified HNSW-indexed vectors across all object types
  summaries        — generated summaries scoped to entity/project/window
  clusters         — thematic groupings of objects
  adjacency_cache  — precomputed cross-type bidirectional neighbourhood

Temporal model (3 layers):
  Event time  — start_time / end_time on events
  Valid time  — valid_from / valid_to on relations and state_assertions
  System time — created_at everywhere

Backward-compat wrappers keep the old upsert_node / add_edge surface
working so main.py can migrate incrementally.
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

ENTITY_TYPES: frozenset[str] = frozenset({
    "person", "place", "org", "object", "concept", "digital_object",
})

EVENT_TYPES: frozenset[str] = frozenset({
    "delivery", "meeting", "conversation", "transaction", "state_change",
})

COMMITMENT_STATUSES: frozenset[str] = frozenset({
    "planned", "confirmed", "completed", "cancelled",
})

COMMITMENT_PRIORITIES: frozenset[str] = frozenset({
    "low", "medium", "high",
})

ALLOWED_RELATION_TYPES: frozenset[str] = frozenset({
    # Entity ↔ Entity
    "knows", "owns", "works_for", "lives_in", "located_in", "part_of",
    # Entity ↔ Event
    "participates_in", "organizes", "affected_by",
    # Entity ↔ Commitment
    "owns_commitment", "responsible_for",
    # Event ↔ Event
    "depends_on", "follows", "blocks",
    # General purpose
    "related_to", "created_by", "used_by", "caused_by", "mentions",
})

# Alias for backward-compat imports
ALLOWED_RELATIONSHIPS = ALLOWED_RELATION_TYPES

# Label → entity type mapping used by migration and upsert_node wrapper
_LABEL_TO_TYPE: dict[str, str] = {
    "person": "person",
    "place": "place",
    "organisation": "org",
    "organization": "org",
    "org": "org",
    "concept": "concept",
    "object": "object",
    "digital_object": "digital_object",
    "event": "concept",   # old Event nodes demoted to concept
    "unknown": "concept",
}

# Old relationship names that need remapping
_REL_REMAP: dict[str, str] = {
    "works_at":    "works_for",
    "instance_of": "part_of",
    "followed_by": "follows",
    "contradicts": "related_to",
    "supports":    "related_to",
}

# ---------------------------------------------------------------------------
# SCHEMA DDL
# ---------------------------------------------------------------------------

_SCHEMA_STMTS = [
    # ── Entities ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS entities (
        id         UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        type       TEXT      NOT NULL,
        name       TEXT      NOT NULL,
        aliases    JSON      DEFAULT '[]',
        attributes JSON      DEFAULT '{}',
        confidence REAL      DEFAULT 1.0,
        source_ids JSON      DEFAULT '[]',
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
    "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",

    # ── Events ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS events (
        id               UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        type             TEXT      NOT NULL,
        title            TEXT      NOT NULL,
        description      TEXT,
        start_time       TIMESTAMP,
        end_time         TIMESTAMP,
        participants     JSON      DEFAULT '[]',
        location_id      UUID,
        related_entities JSON      DEFAULT '[]',
        source_ids       JSON      DEFAULT '[]',
        confidence       REAL      DEFAULT 1.0,
        created_at       TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_title ON events(title)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)",
    "CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time)",

    # ── Commitments ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS commitments (
        id               UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        title            TEXT      NOT NULL,
        description      TEXT,
        owner_id         UUID,
        related_event_id UUID,
        due_time         TIMESTAMP,
        status           TEXT      DEFAULT 'planned',
        priority         TEXT      DEFAULT 'medium',
        constraints      JSON      DEFAULT '{}',
        source_ids       JSON      DEFAULT '[]',
        created_at       TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_commitments_title ON commitments(title)",
    "CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status)",
    "CREATE INDEX IF NOT EXISTS idx_commitments_due ON commitments(due_time)",

    # ── State Assertions ──────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS state_assertions (
        id           UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        subject_id   UUID      NOT NULL,
        subject_type TEXT      NOT NULL,
        predicate    TEXT      NOT NULL,
        value        JSON      NOT NULL,
        valid_from   TIMESTAMP DEFAULT now(),
        valid_to     TIMESTAMP,
        confidence   REAL      DEFAULT 1.0,
        source_ids   JSON      DEFAULT '[]'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_assertions_subject ON state_assertions(subject_id)",

    # ── Sources ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS sources (
        id        UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        type      TEXT      NOT NULL,
        content   TEXT,
        author_id UUID,
        timestamp TIMESTAMP DEFAULT now(),
        metadata  JSON      DEFAULT '{}'
    )
    """,

    # ── Relations ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS relations (
        id         UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        from_id    UUID      NOT NULL,
        from_type  TEXT      NOT NULL,
        to_id      UUID      NOT NULL,
        to_type    TEXT      NOT NULL,
        type       TEXT      NOT NULL,
        valid_from TIMESTAMP,
        valid_to   TIMESTAMP,
        confidence REAL      DEFAULT 1.0,
        source_ids JSON      DEFAULT '[]',
        created_at TIMESTAMP DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_unique ON relations(from_id, to_id, type)",
    "CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id)",
    "CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_id)",
    "CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type)",

    # ── Embeddings ────────────────────────────────────────────────────────
    f"""
    CREATE TABLE IF NOT EXISTS embeddings (
        object_id   TEXT NOT NULL,
        object_type TEXT NOT NULL,
        vector      FLOAT[{EMBEDDING_DIM}],
        model       TEXT DEFAULT 'nomic-embed-text',
        PRIMARY KEY (object_id, object_type)
    )
    """,

    # ── Summaries ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS summaries (
        id           UUID      DEFAULT gen_random_uuid() PRIMARY KEY,
        scope        TEXT      NOT NULL,
        content      TEXT      NOT NULL,
        generated_at TIMESTAMP DEFAULT now(),
        source_ids   JSON      DEFAULT '[]'
    )
    """,

    # ── Clusters ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS clusters (
        id      UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        type    TEXT NOT NULL,
        members JSON DEFAULT '[]'
    )
    """,

    # ── Adjacency Cache ───────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS adjacency_cache (
        node_id       TEXT NOT NULL,
        node_type     TEXT NOT NULL,
        neighbor_id   TEXT NOT NULL,
        neighbor_type TEXT NOT NULL,
        hops          INTEGER NOT NULL,
        PRIMARY KEY (node_id, node_type, neighbor_id, neighbor_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_adj_node ON adjacency_cache(node_id, node_type)",
]

_HNSW_STMT = (
    "CREATE INDEX IF NOT EXISTS embeddings_vector_hnsw "
    "ON embeddings USING HNSW(vector)"
)

# ---------------------------------------------------------------------------
# ADJACENCY CACHE REBUILD
# Bidirectional, up to 2 hops, min hop count per pair.
# Works across all object types via TEXT-cast UUIDs.
# ---------------------------------------------------------------------------

_REFRESH_STMTS = [
    "DELETE FROM adjacency_cache",
    """
    INSERT INTO adjacency_cache (node_id, node_type, neighbor_id, neighbor_type, hops)
    SELECT node_id, node_type, neighbor_id, neighbor_type, MIN(hops)
    FROM (
        SELECT from_id::TEXT AS node_id, from_type AS node_type,
               to_id::TEXT   AS neighbor_id, to_type   AS neighbor_type,
               1             AS hops
          FROM relations
        UNION ALL
        SELECT to_id::TEXT   AS node_id, to_type   AS node_type,
               from_id::TEXT AS neighbor_id, from_type AS neighbor_type,
               1             AS hops
          FROM relations
        UNION ALL
        SELECT r1.from_id::TEXT AS node_id, r1.from_type AS node_type,
               r2.to_id::TEXT   AS neighbor_id, r2.to_type   AS neighbor_type,
               2                AS hops
          FROM relations r1 JOIN relations r2 ON r1.to_id = r2.from_id
        UNION ALL
        SELECT r1.to_id::TEXT   AS node_id, r1.to_type   AS node_type,
               r2.to_id::TEXT   AS neighbor_id, r2.to_type   AS neighbor_type,
               2                AS hops
          FROM relations r1 JOIN relations r2 ON r1.from_id = r2.from_id
         WHERE r1.to_id != r2.to_id
        UNION ALL
        SELECT r1.from_id::TEXT AS node_id, r1.from_type AS node_type,
               r2.from_id::TEXT AS neighbor_id, r2.from_type AS neighbor_type,
               2                AS hops
          FROM relations r1 JOIN relations r2 ON r1.to_id = r2.to_id
         WHERE r1.from_id != r2.from_id
    ) sub
    GROUP BY node_id, node_type, neighbor_id, neighbor_type
    HAVING node_id != neighbor_id
    """,
]

# ---------------------------------------------------------------------------
# GRAPH STORE
# ---------------------------------------------------------------------------


class GraphStore:
    """
    Thread-safe DuckDB knowledge graph implementing the Universal Graph ontology.

    Primary API: upsert_entity, upsert_event, upsert_commitment, add_relation,
                 update_object_embedding, semantic_search, get_active_commitments.

    Legacy API (backward-compat): upsert_node, add_edge, update_node_embedding,
                 query_neighbors, search_nodes_by_name — all delegate to the
                 primary API above.

    All writes hold self._lock (single-threaded DuckDB access).
    Fire-and-forget writes use self._executor (1 worker).
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._conn = duckdb.connect(str(db_path))
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="graph_writer"
        )
        self._init_schema()

    # ------------------------------------------------------------------
    # SCHEMA INITIALISATION + AUTO-MIGRATION
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            try:
                self._conn.execute("INSTALL vss")
            except duckdb.IOException:
                pass
            self._conn.execute("LOAD vss")
            self._conn.execute("SET hnsw_enable_experimental_persistence = true")

            tables = {
                r[0]
                for r in self._conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }

            if "nodes" in tables and "entities" not in tables:
                log.info("Legacy schema detected — running automatic migration…")
                # Create new tables first, then migrate data
                for stmt in _SCHEMA_STMTS:
                    self._conn.execute(stmt)
                self._conn.execute(_HNSW_STMT)
                self._run_migration()
            else:
                for stmt in _SCHEMA_STMTS:
                    self._conn.execute(stmt)
                self._conn.execute(_HNSW_STMT)

            log.debug("Schema initialised")

    def _run_migration(self) -> None:
        """
        Migrate legacy nodes+edges data into the new ontology tables.
        Called while self._lock is held. Leaves old tables (nodes, edges) intact.
        Drops old adjacency_cache so it can be recreated with the new schema.
        """
        conn = self._conn
        # The old adjacency_cache schema lacks node_type/neighbor_type columns.
        # Drop it so _SCHEMA_STMTS can recreate it with the new structure.
        conn.execute("DROP TABLE IF EXISTS adjacency_cache")

        # Step 1 — nodes → entities
        conn.execute("""
            INSERT INTO entities (id, type, name, aliases, attributes,
                                  confidence, source_ids, created_at, updated_at)
            SELECT gen_random_uuid(),
                   lower(label),
                   name,
                   '[]'::JSON,
                   COALESCE(properties, '{}'),
                   1.0,
                   '[]'::JSON,
                   created_at,
                   updated_at
            FROM nodes
        """)

        # Fix type values that don't map cleanly
        conn.execute("""
            UPDATE entities SET type = CASE
                WHEN type = 'event'        THEN 'concept'
                WHEN type = 'organisation' THEN 'org'
                WHEN type = 'organization' THEN 'org'
                WHEN type = 'unknown'      THEN 'concept'
                WHEN type NOT IN ('person','place','org','object',
                                  'concept','digital_object')
                                           THEN 'concept'
                ELSE type
            END
        """)

        # Step 2 — embeddings from nodes.embedding → embeddings table
        conn.execute(f"""
            INSERT INTO embeddings (object_id, object_type, vector, model)
            SELECT e.id::TEXT, 'entity', n.embedding, 'nomic-embed-text'
            FROM nodes n
            JOIN entities e ON n.name = e.name
            WHERE n.embedding IS NOT NULL
        """)

        # Step 3a — edges → relations (normal remaps)
        conn.execute("""
            INSERT INTO relations (id, from_id, from_type, to_id, to_type, type,
                                   valid_from, valid_to, confidence, source_ids, created_at)
            SELECT gen_random_uuid(),
                   ef.id,
                   'entity',
                   et.id,
                   'entity',
                   CASE ed.relationship
                       WHEN 'works_at'    THEN 'works_for'
                       WHEN 'instance_of' THEN 'part_of'
                       WHEN 'followed_by' THEN 'follows'
                       WHEN 'contradicts' THEN 'related_to'
                       WHEN 'supports'    THEN 'related_to'
                       ELSE ed.relationship
                   END,
                   NULL, NULL, 1.0, '[]'::JSON, ed.created_at
            FROM edges ed
            JOIN nodes  nf ON ed.from_id = nf.id
            JOIN nodes  nt ON ed.to_id   = nt.id
            JOIN entities ef ON nf.name  = ef.name
            JOIN entities et ON nt.name  = et.name
            WHERE ed.relationship != 'owned_by'
        """)

        # Step 3b — owned_by: invert direction → owns
        conn.execute("""
            INSERT INTO relations (id, from_id, from_type, to_id, to_type, type,
                                   valid_from, valid_to, confidence, source_ids, created_at)
            SELECT gen_random_uuid(),
                   et.id,   -- swapped: owner is from
                   'entity',
                   ef.id,   -- swapped: owned thing is to
                   'entity',
                   'owns',
                   NULL, NULL, 1.0, '[]'::JSON, ed.created_at
            FROM edges ed
            JOIN nodes  nf ON ed.from_id = nf.id
            JOIN nodes  nt ON ed.to_id   = nt.id
            JOIN entities ef ON nf.name  = ef.name
            JOIN entities et ON nt.name  = et.name
            WHERE ed.relationship = 'owned_by'
        """)

        # Step 4 — rebuild adjacency cache
        for stmt in _REFRESH_STMTS:
            conn.execute(stmt)

        n_ent = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        n_rel = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        n_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        log.info(
            "Migration complete: %d entities, %d relations, %d embeddings",
            n_ent, n_rel, n_emb,
        )

    # ------------------------------------------------------------------
    # ENTITIES
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        name: str,
        type: str,
        aliases: list[str] | None = None,
        attributes: dict | None = None,
        confidence: float = 1.0,
        source_ids: list[str] | None = None,
    ) -> str:
        """
        Insert or update an entity by name (dedup key).
        Returns the entity UUID as a string.
        """
        etype = type.lower().strip()
        if etype not in ENTITY_TYPES:
            etype = "concept"
        key = name.strip()
        aliases_json   = json.dumps(aliases or [])
        attributes_json = json.dumps(attributes or {})
        source_json    = json.dumps(source_ids or [])

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO entities (type, name, aliases, attributes, confidence, source_ids)
                VALUES (?, ?, ?::JSON, ?::JSON, ?, ?::JSON)
                ON CONFLICT (name) DO UPDATE SET
                    type       = EXCLUDED.type,
                    aliases    = EXCLUDED.aliases,
                    attributes = EXCLUDED.attributes,
                    confidence = EXCLUDED.confidence,
                    source_ids = EXCLUDED.source_ids,
                    updated_at = now()
                """,
                [etype, key, aliases_json, attributes_json, confidence, source_json],
            )
            row = self._conn.execute(
                "SELECT id FROM entities WHERE name = ?", [key]
            ).fetchone()
            return str(row[0])

    # ------------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------------

    def upsert_event(
        self,
        title: str,
        type: str,
        description: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        participants: list[str] | None = None,
        location_id: str | None = None,
        related_entities: list[str] | None = None,
        confidence: float = 1.0,
        source_ids: list[str] | None = None,
    ) -> str:
        """
        Insert or update an event by title (dedup key).
        start_time / end_time are ISO-8601 strings or None.
        participants / related_entities are lists of entity UUID strings.
        Returns the event UUID as a string.
        """
        etype = type.lower().strip()
        if etype not in EVENT_TYPES:
            etype = "state_change"
        key = title.strip()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events (type, title, description, start_time, end_time,
                                    participants, location_id, related_entities,
                                    confidence, source_ids)
                VALUES (?, ?, ?, ?::TIMESTAMP, ?::TIMESTAMP,
                        ?::JSON, ?::UUID, ?::JSON, ?, ?::JSON)
                ON CONFLICT (title) DO UPDATE SET
                    type             = EXCLUDED.type,
                    description      = EXCLUDED.description,
                    start_time       = COALESCE(EXCLUDED.start_time, events.start_time),
                    end_time         = COALESCE(EXCLUDED.end_time,   events.end_time),
                    participants     = EXCLUDED.participants,
                    location_id      = COALESCE(EXCLUDED.location_id, events.location_id),
                    related_entities = EXCLUDED.related_entities,
                    confidence       = EXCLUDED.confidence,
                    source_ids       = EXCLUDED.source_ids
                """,
                [
                    etype, key, description,
                    start_time, end_time,
                    json.dumps(participants or []),
                    location_id,
                    json.dumps(related_entities or []),
                    confidence,
                    json.dumps(source_ids or []),
                ],
            )
            row = self._conn.execute(
                "SELECT id FROM events WHERE title = ?", [key]
            ).fetchone()
            return str(row[0])

    # ------------------------------------------------------------------
    # COMMITMENTS
    # ------------------------------------------------------------------

    def upsert_commitment(
        self,
        title: str,
        description: str | None = None,
        owner_id: str | None = None,
        related_event_id: str | None = None,
        due_time: str | None = None,
        status: str = "planned",
        priority: str = "medium",
        constraints: dict | None = None,
        source_ids: list[str] | None = None,
    ) -> str:
        """
        Insert or update a commitment by title (dedup key).
        owner_id and related_event_id are UUID strings or None.
        Returns the commitment UUID as a string.
        """
        s = status.lower().strip() if status else "planned"
        if s not in COMMITMENT_STATUSES:
            s = "planned"
        p = priority.lower().strip() if priority else "medium"
        if p not in COMMITMENT_PRIORITIES:
            p = "medium"
        key = title.strip()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO commitments (title, description, owner_id, related_event_id,
                                         due_time, status, priority, constraints, source_ids)
                VALUES (?, ?, ?::UUID, ?::UUID, ?::TIMESTAMP, ?, ?, ?::JSON, ?::JSON)
                ON CONFLICT (title) DO UPDATE SET
                    description      = EXCLUDED.description,
                    owner_id         = COALESCE(EXCLUDED.owner_id, commitments.owner_id),
                    related_event_id = COALESCE(EXCLUDED.related_event_id, commitments.related_event_id),
                    due_time         = COALESCE(EXCLUDED.due_time, commitments.due_time),
                    status           = EXCLUDED.status,
                    priority         = EXCLUDED.priority,
                    constraints      = EXCLUDED.constraints,
                    source_ids       = EXCLUDED.source_ids
                """,
                [
                    key, description,
                    owner_id, related_event_id,
                    due_time, s, p,
                    json.dumps(constraints or {}),
                    json.dumps(source_ids or []),
                ],
            )
            row = self._conn.execute(
                "SELECT id FROM commitments WHERE title = ?", [key]
            ).fetchone()
            return str(row[0])

    # ------------------------------------------------------------------
    # RELATIONS
    # ------------------------------------------------------------------

    def add_relation(
        self,
        from_id: str,
        from_type: str,
        to_id: str,
        to_type: str,
        relation_type: str,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 1.0,
        source_ids: list[str] | None = None,
    ) -> None:
        """
        Add or update a typed, temporal relation between any two graph objects.
        from_id / to_id are UUID strings. Rebuilds adjacency cache on every write.
        """
        rel = relation_type.lower().strip()
        if rel not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"Relation type '{rel}' not in controlled vocabulary. "
                f"Allowed: {sorted(ALLOWED_RELATION_TYPES)}"
            )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO relations (from_id, from_type, to_id, to_type, type,
                                       valid_from, valid_to, confidence, source_ids)
                VALUES (?::UUID, ?, ?::UUID, ?, ?, ?::TIMESTAMP, ?::TIMESTAMP, ?, ?::JSON)
                ON CONFLICT (from_id, to_id, type) DO UPDATE SET
                    from_type  = EXCLUDED.from_type,
                    to_type    = EXCLUDED.to_type,
                    valid_from = COALESCE(EXCLUDED.valid_from, relations.valid_from),
                    valid_to   = COALESCE(EXCLUDED.valid_to,   relations.valid_to),
                    confidence = EXCLUDED.confidence,
                    source_ids = EXCLUDED.source_ids
                """,
                [
                    from_id, from_type, to_id, to_type, rel,
                    valid_from, valid_to, confidence,
                    json.dumps(source_ids or []),
                ],
            )
            self._refresh_adjacency_cache()

    def _refresh_adjacency_cache(self) -> None:
        """Rebuild full adjacency cache. Must be called while holding self._lock."""
        for stmt in _REFRESH_STMTS:
            self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------

    def update_object_embedding(
        self,
        object_id: str,
        object_type: str,
        embedding: list[float],
        model: str = "nomic-embed-text",
    ) -> None:
        """Store or replace an embedding for any graph object."""
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dimensional, got {len(embedding)}"
            )
        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO embeddings (object_id, object_type, vector, model)
                VALUES (?, ?, ?::FLOAT[{EMBEDDING_DIM}], ?)
                ON CONFLICT (object_id, object_type) DO UPDATE SET
                    vector = EXCLUDED.vector,
                    model  = EXCLUDED.model
                """,
                [object_id, object_type, embedding, model],
            )

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_embedding: list[float],
        top_n: int = 10,
        object_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Return top_n objects most similar to query_embedding by cosine similarity.
        Queries the unified embeddings table; resolves display names via LEFT JOINs.
        object_types filters by 'entity' | 'event' | 'commitment'.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding must be {EMBEDDING_DIM}-dimensional, "
                f"got {len(query_embedding)}"
            )

        type_filter = ""
        params: list = [query_embedding, top_n]
        if object_types:
            placeholders = ", ".join("?" * len(object_types))
            type_filter = f"AND emb.object_type IN ({placeholders})"
            params = [query_embedding] + list(object_types) + [top_n]

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    emb.object_id,
                    emb.object_type,
                    COALESCE(e.name, ev.title, c.title, emb.object_id) AS display_name,
                    COALESCE(e.type,  ev.type,  c.status)              AS subtype,
                    array_cosine_similarity(
                        emb.vector, ?::FLOAT[{EMBEDDING_DIM}]
                    ) AS score
                FROM embeddings emb
                LEFT JOIN entities    e  ON emb.object_id = e.id::TEXT
                                        AND emb.object_type = 'entity'
                LEFT JOIN events      ev ON emb.object_id = ev.id::TEXT
                                        AND emb.object_type = 'event'
                LEFT JOIN commitments c  ON emb.object_id = c.id::TEXT
                                        AND emb.object_type = 'commitment'
                WHERE emb.vector IS NOT NULL
                {type_filter}
                ORDER BY score DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            {
                "object_id":    r[0],
                "object_type":  r[1],
                "display_name": r[2],
                "subtype":      r[3],
                "score":        float(r[4]),
            }
            for r in rows
        ]

    def _query_neighbors_by_id(
        self,
        node_id: str,
        node_type: str,
        hops: int = 2,
    ) -> list[dict]:
        """
        Return all objects reachable from (node_id, node_type) within N hops.
        Traversal is bidirectional via the adjacency cache.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT ac.neighbor_id, ac.neighbor_type, ac.hops,
                       COALESCE(e.name, ev.title, c.title) AS display_name,
                       COALESCE(e.type, ev.type, c.status) AS subtype
                FROM adjacency_cache ac
                LEFT JOIN entities    e  ON ac.neighbor_id = e.id::TEXT
                                        AND ac.neighbor_type = 'entity'
                LEFT JOIN events      ev ON ac.neighbor_id = ev.id::TEXT
                                        AND ac.neighbor_type = 'event'
                LEFT JOIN commitments c  ON ac.neighbor_id = c.id::TEXT
                                        AND ac.neighbor_type = 'commitment'
                WHERE ac.node_id = ? AND ac.node_type = ? AND ac.hops <= ?
                ORDER BY ac.hops, display_name
                """,
                [node_id, node_type, hops],
            ).fetchall()
        return [
            {
                "object_id":    r[0],
                "object_type":  r[1],
                "hops":         r[2],
                "display_name": r[3] or r[0],
                "subtype":      r[4],
            }
            for r in rows
        ]

    def search_by_name(self, fragment: str, limit: int = 5) -> list[dict]:
        """
        Case-insensitive substring search across entities, events, and commitments.
        Returns list of dicts: {object_id, object_type, subtype, display_name}.
        """
        pattern = f"%{fragment.strip()}%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id::TEXT, 'entity' AS object_type, type AS subtype, name AS display_name
                  FROM entities WHERE name ILIKE ?
                UNION ALL
                SELECT id::TEXT, 'event', type, title
                  FROM events WHERE title ILIKE ?
                UNION ALL
                SELECT id::TEXT, 'commitment', status, title
                  FROM commitments WHERE title ILIKE ?
                LIMIT ?
                """,
                [pattern, pattern, pattern, limit],
            ).fetchall()
        return [
            {
                "object_id":    r[0],
                "object_type":  r[1],
                "subtype":      r[2],
                "display_name": r[3],
            }
            for r in rows
        ]

    def get_active_commitments(
        self,
        status_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Return commitments ordered by due_time.
        status_filter defaults to ['planned', 'confirmed'].
        """
        statuses = status_filter or ["planned", "confirmed"]
        placeholders = ", ".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.id::TEXT, c.title, c.description, c.status, c.priority,
                       c.due_time, c.owner_id::TEXT, e.name AS owner_name,
                       c.related_event_id::TEXT
                  FROM commitments c
                  LEFT JOIN entities e ON c.owner_id = e.id
                 WHERE c.status IN ({placeholders})
                 ORDER BY c.due_time NULLS LAST, c.priority DESC
                """,
                statuses,
            ).fetchall()
        return [
            {
                "id":          r[0],
                "title":       r[1],
                "description": r[2],
                "status":      r[3],
                "priority":    r[4],
                "due_time":    str(r[5]) if r[5] else None,
                "owner_id":    r[6],
                "owner_name":  r[7],
                "event_id":    r[8],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        """Return row counts for all primary tables."""
        with self._lock:
            return {
                "entities":    self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                "events":      self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "commitments": self._conn.execute("SELECT COUNT(*) FROM commitments").fetchone()[0],
                "relations":   self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
                "embeddings":  self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
            }

    # ------------------------------------------------------------------
    # ASYNC / FIRE-AND-FORGET WRAPPERS
    # ------------------------------------------------------------------

    async def async_upsert_entity(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.upsert_entity, *args, **kwargs)

    async def async_upsert_event(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.upsert_event, *args, **kwargs)

    async def async_upsert_commitment(self, *args, **kwargs) -> str:
        return await asyncio.to_thread(self.upsert_commitment, *args, **kwargs)

    async def async_add_relation(self, *args, **kwargs) -> None:
        await asyncio.to_thread(self.add_relation, *args, **kwargs)

    def submit_upsert_entity(self, *args, **kwargs) -> Future:
        return self._executor.submit(self.upsert_entity, *args, **kwargs)

    def submit_upsert_event(self, *args, **kwargs) -> Future:
        return self._executor.submit(self.upsert_event, *args, **kwargs)

    def submit_upsert_commitment(self, *args, **kwargs) -> Future:
        return self._executor.submit(self.upsert_commitment, *args, **kwargs)

    def submit_add_relation(self, *args, **kwargs) -> Future:
        return self._executor.submit(self.add_relation, *args, **kwargs)

    # ------------------------------------------------------------------
    # BACKWARD-COMPAT WRAPPERS (legacy upsert_node / add_edge surface)
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        name: str,
        label: str,
        properties: dict | None = None,
    ) -> str:
        """Legacy wrapper → upsert_entity(). Returns UUID string."""
        etype = _LABEL_TO_TYPE.get(label.lower().strip(), "concept")
        return self.upsert_entity(name, etype, attributes=properties)

    def add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        """
        Legacy wrapper → add_relation().
        Auto-creates stub entities if endpoints are absent.
        weight maps to confidence.
        """
        rel = relationship.lower().strip()
        rel = _REL_REMAP.get(rel, rel)
        if rel not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"Relationship '{rel}' not in controlled vocabulary. "
                f"Allowed: {sorted(ALLOWED_RELATION_TYPES)}"
            )
        from_id = self.upsert_entity(from_name.strip(), "concept")
        to_id   = self.upsert_entity(to_name.strip(), "concept")
        self.add_relation(from_id, "entity", to_id, "entity", rel,
                          confidence=float(weight))

    def update_node_embedding(self, name: str, embedding: list[float]) -> None:
        """Legacy wrapper → update_object_embedding()."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM entities WHERE name = ?", [name.strip()]
            ).fetchone()
        if row:
            self.update_object_embedding(str(row[0]), "entity", embedding)

    def query_neighbors(self, node_name: str, hops: int = 2) -> list[dict]:
        """Legacy wrapper → _query_neighbors_by_id(). Assumes entity type."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM entities WHERE name = ?", [node_name.strip()]
            ).fetchone()
        if not row:
            return []
        neighbors = self._query_neighbors_by_id(str(row[0]), "entity", hops)
        return [
            {
                "name":       n["display_name"],
                "label":      n["subtype"] or n["object_type"],
                "properties": {},
                "hops":       n["hops"],
            }
            for n in neighbors
        ]

    def search_nodes_by_name(self, fragment: str, limit: int = 5) -> list[dict]:
        """Legacy wrapper → search_by_name(). Returns old-style dicts."""
        return [
            {"name": r["display_name"], "label": r["subtype"] or r["object_type"], "properties": {}}
            for r in self.search_by_name(fragment, limit)
        ]

    def submit_upsert_node(
        self, name: str, label: str, properties: dict | None = None
    ) -> Future:
        return self._executor.submit(self.upsert_node, name, label, properties)

    def submit_add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> Future:
        return self._executor.submit(self.add_edge, from_name, to_name, relationship, weight)

    async def async_upsert_node(
        self, name: str, label: str, properties: dict | None = None
    ) -> str:
        return await asyncio.to_thread(self.upsert_node, name, label, properties)

    async def async_add_edge(
        self,
        from_name: str,
        to_name: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        await asyncio.to_thread(self.add_edge, from_name, to_name, relationship, weight)

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

        # ── entities ──────────────────────────────────────────────────
        id_boris  = gs.upsert_entity("Boris",  "person",  attributes={"city": "London"})
        id_london = gs.upsert_entity("London", "place")
        id_lyon   = gs.upsert_entity("Lyon",   "place")
        id_desk   = gs.upsert_entity("Desk",   "object")

        id_boris2 = gs.upsert_entity("Boris", "person", attributes={"city": "Paris"})
        check(id_boris == id_boris2, "entity dedup: same UUID on second upsert")

        row = gs._conn.execute(
            "SELECT attributes FROM entities WHERE name = 'Boris'"
        ).fetchone()
        attrs = json.loads(row[0])
        check(attrs.get("city") == "Paris", "entity dedup: attributes updated on conflict")

        # ── events ────────────────────────────────────────────────────
        id_delivery = gs.upsert_event(
            "Desk delivery Mar 23",
            "delivery",
            description="IKEA desk delivered to Jarente flat",
            start_time="2025-03-23 09:00:00",
            participants=[id_boris],
            location_id=id_lyon,
        )
        check(id_delivery is not None, "event upserted")

        id_delivery2 = gs.upsert_event("Desk delivery Mar 23", "delivery")
        check(id_delivery == id_delivery2, "event dedup: same UUID on second upsert")

        # ── commitments ───────────────────────────────────────────────
        id_c = gs.upsert_commitment(
            "Be present for desk delivery",
            owner_id=id_boris,
            related_event_id=id_delivery,
            due_time="2025-03-23 09:00:00",
            status="planned",
            priority="high",
        )
        check(id_c is not None, "commitment upserted")

        active = gs.get_active_commitments()
        check(len(active) >= 1, f"get_active_commitments returns ≥1, got {len(active)}")
        check(active[0]["priority"] == "high", "commitment priority correct")

        # ── relations ─────────────────────────────────────────────────
        gs.add_relation(id_boris, "entity", id_london, "entity", "lives_in")
        gs.add_relation(id_boris, "entity", id_delivery, "event", "participates_in")
        gs.add_relation(id_boris, "entity", id_c, "commitment", "owns_commitment")

        nb1 = gs._query_neighbors_by_id(id_boris, "entity", hops=1)
        check(len(nb1) >= 2, f"1-hop neighbors of Boris: expected ≥2, got {len(nb1)}")

        # ── search_by_name (cross-type) ───────────────────────────────
        hits = gs.search_by_name("desk", limit=10)
        hit_types = {h["object_type"] for h in hits}
        check("event" in hit_types or "entity" in hit_types,
              "search_by_name finds cross-type results")

        # ── embeddings & semantic_search ──────────────────────────────
        rng = random.Random(42)

        def rand_unit_vec() -> list[float]:
            v = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
            mag = sum(x ** 2 for x in v) ** 0.5
            return [x / mag for x in v]

        for oid, otype in [(id_boris, "entity"), (id_london, "entity"),
                           (id_delivery, "event"), (id_c, "commitment")]:
            gs.update_object_embedding(oid, otype, rand_unit_vec())

        results = gs.semantic_search(rand_unit_vec(), top_n=4)
        check(len(results) == 4, f"semantic_search returns 4 results, got {len(results)}")
        check(results[0]["score"] >= results[-1]["score"], "results ordered by score desc")
        result_types = {r["object_type"] for r in results}
        check(len(result_types) > 1, "semantic_search spans multiple object types")

        # ── HNSW index exists ─────────────────────────────────────────
        idx_row = gs._conn.execute(
            "SELECT index_name FROM duckdb_indexes() "
            "WHERE index_name = 'embeddings_vector_hnsw'"
        ).fetchone()
        check(idx_row is not None, "HNSW index on embeddings table exists")

        # ── stats ─────────────────────────────────────────────────────
        s = gs.stats()
        check(s["entities"] >= 4,    f"stats: entities ≥4, got {s['entities']}")
        check(s["events"] >= 1,      f"stats: events ≥1, got {s['events']}")
        check(s["commitments"] >= 1, f"stats: commitments ≥1, got {s['commitments']}")
        check(s["relations"] >= 3,   f"stats: relations ≥3, got {s['relations']}")
        check(s["embeddings"] >= 4,  f"stats: embeddings ≥4, got {s['embeddings']}")

        # ── backward-compat wrappers ──────────────────────────────────
        id_compat = gs.upsert_node("Ada Lovelace", "Person", {"field": "maths"})
        check(id_compat is not None, "upsert_node backward-compat wrapper works")

        gs.add_edge("Ada Lovelace", "London", "lives_in", weight=0.8)
        nb_compat = gs.query_neighbors("Ada Lovelace", hops=1)
        check(len(nb_compat) >= 1, "query_neighbors backward-compat wrapper works")

        try:
            gs.add_edge("Boris", "London", "invented", weight=1.0)
            check(False, "ValueError raised for unknown relationship")
        except ValueError:
            check(True, "ValueError raised for unknown relationship")

        # ── fire-and-forget ───────────────────────────────────────────
        f = gs.submit_upsert_entity("Lyon", "place")
        f.result()
        row = gs._conn.execute("SELECT id FROM entities WHERE name = 'Lyon'").fetchone()
        check(row is not None, "fire-and-forget submit_upsert_entity works")

    test_db.unlink(missing_ok=True)

    if errors:
        log.error("%d check(s) failed.", len(errors))
        sys.exit(1)

    print("All checks passed.")
    sys.exit(0)
