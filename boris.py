"""
boris.py — Visualise the Living Memory DuckDB knowledge graph with pyvis.

Queries entities, events, and commitments from the Universal Graph schema.
Falls back to the legacy nodes/edges tables if the new schema is absent.

Run:
    cd "/Users/borisdadvisard/Documents/ClaudeCode/Living Memory"
    source venv/bin/activate
    pip install pyvis pandas          # first time only
    venv/bin/python boris.py
    open graph.html
"""

'''also install and run Harlequin to edit the DB
cd "/Users/borisdadvisard/Documents/ClaudeCode/Living Memory"
source venv/bin/activate
pip install harlequin
harlequin graph.db --init-path init.sql
'''

import json
import duckdb
from pyvis.network import Network

# ── 1. Connect & load VSS (required to open graph.db) ───────────────────────
con = duckdb.connect("graph.db")
try:
    con.execute("INSTALL vss")
except duckdb.IOException:
    pass
con.execute("LOAD vss")
con.execute("SET hnsw_enable_experimental_persistence = true")

# ── 2. Detect schema version ─────────────────────────────────────────────────
tables = {
    r[0]
    for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
}
use_new_schema = "entities" in tables

# ── 3. Query nodes & edges ────────────────────────────────────────────────────
if use_new_schema:
    # Universal Graph schema: UNION entities + events + commitments
    nodes_df = con.sql("""
        SELECT id::TEXT AS id, name AS label_name, 'entity:'||type AS type_tag,
               attributes::TEXT AS props_json
        FROM entities
        UNION ALL
        SELECT id::TEXT, title, 'event:'||type,
               json_object('start_time', CAST(start_time AS TEXT),
                           'description', COALESCE(description, ''))
        FROM events
        UNION ALL
        SELECT id::TEXT, title, 'commitment:'||status,
               json_object('priority', priority, 'due_time', CAST(due_time AS TEXT))
        FROM commitments
    """).df()

    edges_df = con.sql("""
        SELECT
            r.from_id::TEXT AS from_id,
            r.to_id::TEXT   AS to_id,
            r.type          AS relationship,
            r.confidence    AS weight
        FROM relations r
    """).df()

    print(f"[Universal Graph schema]")
else:
    # Legacy schema fallback
    nodes_df = con.sql("""
        SELECT id::TEXT AS id, name AS label_name,
               'entity:'||lower(label) AS type_tag,
               properties::TEXT AS props_json
        FROM nodes
    """).df()

    edges_df = con.sql("""
        SELECT fn.id::TEXT AS from_id, tn.id::TEXT AS to_id,
               e.relationship, e.weight
        FROM edges e
        JOIN nodes fn ON e.from_id = fn.id
        JOIN nodes tn ON e.to_id   = tn.id
    """).df()

    print(f"[Legacy schema — run migrate.py to upgrade]")

print(f"Nodes: {len(nodes_df)}  |  Edges: {len(edges_df)}")
if not nodes_df.empty:
    print(nodes_df[["id", "label_name", "type_tag"]].to_string(index=False))

# ── 4. Colour map by type_tag ─────────────────────────────────────────────────
COLORS: dict[str, str] = {
    # Entities
    "entity:person":         "#7F77DD",
    "entity:place":          "#1D9E75",
    "entity:org":            "#4AABDB",
    "entity:object":         "#EF9F27",
    "entity:concept":        "#EF9F27",
    "entity:digital_object": "#B0D0FF",
    # Events
    "event:delivery":        "#E05C5C",
    "event:meeting":         "#E08C5C",
    "event:conversation":    "#E0B05C",
    "event:transaction":     "#C05CE0",
    "event:state_change":    "#D0D050",
    # Commitments
    "commitment:planned":    "#FFB347",
    "commitment:confirmed":  "#FFA500",
    "commitment:completed":  "#90EE90",
    "commitment:cancelled":  "#AAAAAA",
    # Legacy fallbacks
    "entity:event":          "#E05C5C",
    "entity:unknown":        "#AAAAAA",
}

# ── 5. Build the pyvis network ────────────────────────────────────────────────
net = Network(
    height="750px", width="100%",
    bgcolor="#1a1a2e", font_color="#e0e0e0",
    notebook=False,
    directed=True,
)

for _, n in nodes_df.iterrows():
    type_tag = str(n["type_tag"]).lower()
    color = COLORS.get(type_tag, "#888888")

    # Shape differs by object category
    category = type_tag.split(":")[0] if ":" in type_tag else "entity"
    shape = {"entity": "dot", "event": "diamond", "commitment": "star"}.get(
        category, "dot"
    )

    props: dict = {}
    if n["props_json"]:
        try:
            props = json.loads(n["props_json"])
        except (ValueError, TypeError):
            pass
    props_str = "\n".join(f"{k}: {v}" for k, v in props.items() if v and v != "None")
    tooltip = f"{n['label_name']} [{n['type_tag']}]"
    if props_str:
        tooltip += f"\n{props_str}"

    net.add_node(
        n["id"],
        label=n["label_name"],
        color=color,
        size=20 if category == "commitment" else 18,
        shape=shape,
        title=tooltip,
        font={"size": 14},
    )

# Build id → node_name lookup for edge rendering
id_to_name = dict(zip(nodes_df["id"], nodes_df["label_name"]))

for _, e in edges_df.iterrows():
    if e["from_id"] not in id_to_name or e["to_id"] not in id_to_name:
        continue
    net.add_edge(
        e["from_id"],
        e["to_id"],
        title=e["relationship"],
        label=e["relationship"],
        width=max(1.0, float(e["weight"]) * 3),
        color="rgba(180,180,200,0.5)",
        font={"size": 11, "color": "#cccccc"},
        arrows="to",
    )

# ── 6. Physics layout ─────────────────────────────────────────────────────────
net.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "springLength": 140,
      "springConstant": 0.05
    },
    "solver": "forceAtlas2Based",
    "stabilization": { "iterations": 150 }
  },
  "edges": {
    "smooth": { "type": "dynamic" }
  }
}
""")

# ── 7. Export ──────────────────────────────────────────────────────────────────
out = "Viz/graph.html"
net.show(out, notebook=False)
print(f"Saved → {out}   (run: open {out})")
