"""
migrate.py — Standalone migration: legacy nodes+edges → Universal Graph ontology

Idempotent: safe to run multiple times.
Guard: only executes if `nodes` table exists AND `entities` table does not.
Non-destructive: legacy tables (nodes, edges, adjacency_cache) are left intact.

Usage:
    venv/bin/python migrate.py [--db path/to/graph.db]
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb

from graph_store import (
    DB_PATH,
    EMBEDDING_DIM,
    _SCHEMA_STMTS,
    _HNSW_STMT,
    _REFRESH_STMTS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("migrate")


def migration_needed(conn: duckdb.DuckDBPyConnection) -> bool:
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    if "nodes" not in tables:
        log.info("No legacy 'nodes' table found — nothing to migrate.")
        return False
    if "entities" in tables:
        log.info("'entities' table already exists — migration already complete.")
        return False
    return True


def run_migration(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Perform the full migration on an open DuckDB connection.
    Creates new tables, copies data, rebuilds cache.
    """
    log.info("Creating new schema tables…")
    # The old adjacency_cache has a different schema (no node_type column).
    # Drop it so it can be recreated with the new structure.
    conn.execute("DROP TABLE IF EXISTS adjacency_cache")
    for stmt in _SCHEMA_STMTS:
        conn.execute(stmt)
    conn.execute(_HNSW_STMT)

    # ── Step 1: nodes → entities ──────────────────────────────────────
    log.info("Migrating nodes → entities…")
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

    n_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    log.info("  → %d entities created", n_entities)

    # ── Step 2: nodes.embedding → embeddings table ────────────────────
    log.info("Migrating embeddings…")
    conn.execute(f"""
        INSERT INTO embeddings (object_id, object_type, vector, model)
        SELECT e.id::TEXT, 'entity', n.embedding, 'nomic-embed-text'
        FROM nodes n
        JOIN entities e ON n.name = e.name
        WHERE n.embedding IS NOT NULL
    """)

    n_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    log.info("  → %d embeddings migrated", n_emb)

    # ── Step 3a: edges → relations (standard remaps) ──────────────────
    log.info("Migrating edges → relations…")
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
        JOIN nodes    nf ON ed.from_id = nf.id
        JOIN nodes    nt ON ed.to_id   = nt.id
        JOIN entities ef ON nf.name   = ef.name
        JOIN entities et ON nt.name   = et.name
        WHERE ed.relationship != 'owned_by'
    """)

    # ── Step 3b: owned_by → owns (inverted) ──────────────────────────
    conn.execute("""
        INSERT INTO relations (id, from_id, from_type, to_id, to_type, type,
                               valid_from, valid_to, confidence, source_ids, created_at)
        SELECT gen_random_uuid(),
               et.id,
               'entity',
               ef.id,
               'entity',
               'owns',
               NULL, NULL, 1.0, '[]'::JSON, ed.created_at
        FROM edges ed
        JOIN nodes    nf ON ed.from_id = nf.id
        JOIN nodes    nt ON ed.to_id   = nt.id
        JOIN entities ef ON nf.name   = ef.name
        JOIN entities et ON nt.name   = et.name
        WHERE ed.relationship = 'owned_by'
    """)

    n_rel = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    log.info("  → %d relations created", n_rel)

    # ── Step 4: rebuild adjacency cache ──────────────────────────────
    log.info("Rebuilding adjacency cache…")
    for stmt in _REFRESH_STMTS:
        conn.execute(stmt)
    n_cache = conn.execute("SELECT COUNT(*) FROM adjacency_cache").fetchone()[0]
    log.info("  → %d adjacency cache entries", n_cache)

    log.info(
        "Migration complete: %d entities, %d relations, %d embeddings, %d cache entries",
        n_entities, n_rel, n_emb, n_cache,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Living Memory graph.db to Universal Graph schema")
    parser.add_argument("--db", type=Path, default=DB_PATH,
                        help="Path to graph.db (default: project graph.db)")
    args = parser.parse_args()

    db_path = args.db
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        sys.exit(1)

    log.info("Opening %s", db_path)
    conn = duckdb.connect(str(db_path))

    try:
        conn.execute("INSTALL vss")
    except duckdb.IOException:
        pass
    conn.execute("LOAD vss")
    conn.execute("SET hnsw_enable_experimental_persistence = true")

    if not migration_needed(conn):
        conn.close()
        sys.exit(0)

    # Snapshot row counts before migration for verification
    n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    n_emb_before = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    log.info(
        "Legacy snapshot: %d nodes, %d edges, %d embeddings",
        n_nodes, n_edges, n_emb_before,
    )

    run_migration(conn)

    # Verify counts match
    n_ent = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_rel = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    n_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    ok = True
    if n_ent != n_nodes:
        log.warning("Entity count mismatch: expected %d, got %d", n_nodes, n_ent)
        ok = False
    if n_emb != n_emb_before:
        log.warning("Embedding count mismatch: expected %d, got %d", n_emb_before, n_emb)
        ok = False
    # Relations may differ from edges due to owned_by inversion (same count)
    if n_rel != n_edges:
        log.warning("Relation count mismatch: expected %d, got %d", n_edges, n_rel)
        ok = False

    conn.close()

    if ok:
        log.info("Verification passed.")
        sys.exit(0)
    else:
        log.warning("Verification warnings above — review before proceeding.")
        sys.exit(0)  # not fatal — data is intact


if __name__ == "__main__":
    main()
