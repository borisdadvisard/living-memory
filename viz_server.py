"""
viz_server.py — Lightweight HTTP server for live Living Memory graph visualization.

Serves a self-contained Vis.js page that polls GET /graph every 3 seconds and
applies diff updates to the DataSets, preserving node positions as the graph grows.

Usage from main.py:
    import viz_server
    viz_server.start(gs)   # gs is a live GraphStore instance
"""

import json
import logging
import os
import pathlib
import threading
import time
import urllib.request
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent
PORT = int(os.getenv("LM_VIZ_PORT", "5050"))

_gs                = None   # set by start()
_handle_chat_turn  = None   # set by start(); callable(message) → SSE generator


# ---------------------------------------------------------------------------
# GRAPH DATA QUERY
# ---------------------------------------------------------------------------

def _build_graph_data() -> dict:
    """Return current graph state as {nodes, edges} suitable for Vis.js."""
    with _gs._lock:
        nodes_rows = _gs._conn.execute("""
            SELECT id::TEXT, name,  'entity:'     || type   FROM entities
            UNION ALL
            SELECT id::TEXT, title, 'event:'      || type   FROM events
            UNION ALL
            SELECT id::TEXT, title, 'commitment:' || status FROM commitments
        """).fetchall()

        edges_rows = _gs._conn.execute("""
            SELECT from_id::TEXT, to_id::TEXT, type,
                   strftime(created_at, '%Y-%m-%dT%H:%M:%S') AS created_at
            FROM relations
        """).fetchall()

    degree: Counter = Counter()
    for from_id, to_id, *_ in edges_rows:
        degree[from_id] += 1
        degree[to_id] += 1

    node_ids = {r[0] for r in nodes_rows}

    nodes = [
        {
            "id":       r[0],
            "label":    r[1],
            "type_tag": r[2].lower(),
            "degree":   degree.get(r[0], 0),
        }
        for r in nodes_rows
    ]
    edges = [
        {"id": f"e{i}", "from": r[0], "to": r[1], "type": r[2], "created_at": r[3]}
        for i, r in enumerate(edges_rows)
        if r[0] in node_ids and r[1] in node_ids
    ]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# HTML — live polling version of the graph visualization
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Living Memory</title>
  <link rel="stylesheet" href="lib/vis-9.1.2/vis-network.css">
  <script src="lib/vis-9.1.2/vis-network.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body { background: #09090f; overflow: hidden; }

    #graph {
      width: 100vw;
      height: 100vh;
      background-color: #09090f;
      background-image:
        linear-gradient(rgba(100, 130, 210, 0.032) 1px, transparent 1px),
        linear-gradient(90deg, rgba(100, 130, 210, 0.032) 1px, transparent 1px);
      background-size: 48px 48px;
    }

    #reset-hint {
      position: fixed;
      bottom: 26px;
      left: 50%;
      transform: translateX(-50%);
      color: rgba(127, 255, 212, 0);
      font-family: 'DM Mono', 'Courier New', monospace;
      font-size: 11px;
      letter-spacing: 0.2em;
      text-transform: lowercase;
      transition: color 0.7s ease;
      pointer-events: none;
      z-index: 10;
    }
    #reset-hint.visible { color: rgba(127, 255, 212, 0.42); }

    #meta {
      position: fixed;
      top: 18px;
      right: 22px;
      color: rgba(127, 255, 212, 0.16);
      font-family: 'DM Mono', 'Courier New', monospace;
      font-size: 10px;
      letter-spacing: 0.24em;
      z-index: 10;
      user-select: none;
    }
  </style>
</head>
<body>
  <div id="graph"></div>
  <div id="reset-hint">click anywhere to reset</div>
  <div id="meta">connecting\u2026</div>

  <script>
    // \u2500\u2500 Colour palette \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const TYPE_COLORS = {
      'entity:person':         '#7fffd4',
      'entity:place':          '#6a8fd8',
      'entity:org':            '#9b8ed6',
      'entity:object':         '#5ab4d6',
      'entity:concept':        '#a3d9b1',
      'entity:digital_object': '#7ecfeb',
      'event:delivery':        '#e07c7c',
      'event:meeting':         '#e07c7c',
      'event:conversation':    '#e07c7c',
      'event:transaction':     '#e07c7c',
      'event:state_change':    '#e07c7c',
      'commitment:planned':    '#d4a85a',
      'commitment:confirmed':  '#d4a85a',
      'commitment:completed':  '#7fcc7f',
      'commitment:cancelled':  '#55566a',
    };
    const DEFAULT_COLOR = '#8888aa';
    const nodeColor = t => TYPE_COLORS[t] || DEFAULT_COLOR;

    // \u2500\u2500 DataSets \u2014 populated via /graph polling \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const MIN_SIZE = 9, MAX_SIZE = 32;
    const visNodes = new vis.DataSet([]);
    const visEdges = new vis.DataSet([]);
    let particles = [];
    let maxDegree = 1;

    // \u2500\u2500 Network \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const container = document.getElementById('graph');
    const network   = new vis.Network(container, { nodes: visNodes, edges: visEdges }, {
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -80,
          springLength: 150,
          springConstant: 0.04,
          damping: 0.6,
        },
        stabilization: { iterations: 300, updateInterval: 10 },
        maxVelocity: 50,
        minVelocity: 0.1,
      },
      interaction: {
        hover: true,
        tooltipDelay: 9999999,
        hideEdgesOnDrag: false,
        zoomView: true,
        dragNodes: true,
      },
      nodes: { borderWidth: 0, borderWidthSelected: 2 },
      edges: { smooth: { type: 'dynamic' }, selectionWidth: 0 },
    });

    // \u2500\u2500 Stabilisation \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    let stabilised = false;
    network.on('stabilizationIterationsDone', () => {
      stabilised = true;
      network.setOptions({ physics: { enabled: false } });
    });

    // \u2500\u2500 Hover: show / hide label \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    let focusedNode = null;

    network.on('hoverNode', ({ node }) => {
      if (focusedNode !== null) return;
      const n = visNodes.get(node);
      visNodes.update({ id: node, label: n._label,
        font: { size: 13, color: n._color, face: 'DM Mono, Courier New, monospace', strokeWidth: 0 } });
    });

    network.on('blurNode', ({ node }) => {
      if (focusedNode !== null) return;
      visNodes.update({ id: node, label: '', font: { size: 0 } });
    });

    // \u2500\u2500 Click: focus / reset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    network.on('click', params => {
      if (params.nodes.length > 0) {
        const id = params.nodes[0];
        focusedNode === id ? resetFocus() : applyFocus(id);
      } else if (params.edges.length === 0) {
        resetFocus();
      }
    });

    function applyFocus(nodeId) {
      focusedNode = nodeId;
      const neighbours     = new Set(network.getConnectedNodes(nodeId));
      neighbours.add(nodeId);
      const connectedEdges = new Set(network.getConnectedEdges(nodeId));

      visNodes.update(visNodes.getIds().map(id => {
        const n    = visNodes.get(id);
        const show = neighbours.has(id);
        return {
          id,
          hidden: !show,
          label:  show ? n._label : '',
          font: { size: show ? 13 : 0, color: n._color,
                  face: 'DM Mono, Courier New, monospace', strokeWidth: 0 },
        };
      }));
      visEdges.update(visEdges.getIds().map(id => ({
        id, hidden: !connectedEdges.has(id),
      })));
      document.getElementById('reset-hint').classList.add('visible');
    }

    function resetFocus() {
      if (focusedNode === null) return;
      focusedNode = null;
      visNodes.update(visNodes.getIds().map(id => ({
        id, hidden: false, label: '', font: { size: 0 },
      })));
      visEdges.update(visEdges.getIds().map(id => ({ id, hidden: false })));
      document.getElementById('reset-hint').classList.remove('visible');
    }

    // \u2500\u2500 Vis.js object builders \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    function buildVisNode(n) {
      const c    = nodeColor(n.type_tag);
      const size = MIN_SIZE + (n.degree / maxDegree) * (MAX_SIZE - MIN_SIZE);
      return {
        id: n.id, _label: n.label, _color: c,
        label: '',
        color: {
          background: c, border: c,
          highlight: { background: '#ffffff', border: c },
          hover:      { background: c, border: '#ffffff' },
        },
        size, shape: 'dot', borderWidth: 0, borderWidthSelected: 2,
        font: { size: 0, color: c, face: 'DM Mono, Courier New, monospace', strokeWidth: 0 },
        shadow: { enabled: true, color: c + '55', size: 14, x: 0, y: 0 },
      };
    }

    function buildVisEdge(e) {
      return {
        id: e.id, from: e.from, to: e.to, _type: e.type,
        color: {
          color:     'rgba(110, 120, 175, 0.18)',
          highlight: 'rgba(200, 215, 255, 0.55)',
          hover:     'rgba(200, 215, 255, 0.40)',
        },
        width: 1, selectionWidth: 0,
        smooth: { type: 'dynamic' },
        arrows: { to: { enabled: true, scaleFactor: 0.35 } },
      };
    }

    // \u2500\u2500 Live graph polling \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    async function refreshGraph() {
      let data;
      try {
        const resp = await fetch('/graph');
        data = await resp.json();
      } catch(e) {
        return;  // server not yet ready; retry on next interval
      }

      maxDegree = Math.max(...data.nodes.map(n => n.degree), 1);

      const existingNodeIds = new Set(visNodes.getIds());
      const existingEdgeIds = new Set(visEdges.getIds());
      const newNodeIds      = new Set(data.nodes.map(n => n.id));
      const newEdgeIds      = new Set(data.edges.map(e => e.id));

      const toRemoveNodes = [...existingNodeIds].filter(id => !newNodeIds.has(id));
      const toAddNodes    = data.nodes.filter(n => !existingNodeIds.has(n.id));
      const toUpdateNodes = data.nodes.filter(n =>  existingNodeIds.has(n.id));
      const toRemoveEdges = [...existingEdgeIds].filter(id => !newEdgeIds.has(id));
      const toAddEdges    = data.edges.filter(e => !existingEdgeIds.has(e.id));

      if (toRemoveNodes.length) visNodes.remove(toRemoveNodes);
      if (toAddNodes.length)    visNodes.add(toAddNodes.map(buildVisNode));
      if (toUpdateNodes.length) visNodes.update(toUpdateNodes.map(buildVisNode));

      if (toRemoveEdges.length) {
        visEdges.remove(toRemoveEdges);
        const gone = new Set(toRemoveEdges);
        particles = particles.filter(p => !gone.has(p.edgeId));
      }
      if (toAddEdges.length) {
        visEdges.add(toAddEdges.map(buildVisEdge));
        toAddEdges.forEach(e => {
          const src = visNodes.get(e.from);
          const c   = src ? src._color : DEFAULT_COLOR;
          particles.push({ fromId: e.from, toId: e.to, phase: Math.random(), color: c, edgeId: e.id });
          if (data.edges.length < 60 || Math.random() > 0.45)
            particles.push({ fromId: e.from, toId: e.to, phase: Math.random(), color: c, edgeId: e.id });
        });
      }

      // Re-enable physics so new nodes settle into position
      if (toAddNodes.length > 0) {
        stabilised = false;
        network.setOptions({ physics: { enabled: true } });
      }

      document.getElementById('meta').textContent =
        data.nodes.length + ' nodes  \u00b7  ' + data.edges.length + ' edges';
    }

    refreshGraph();
    setInterval(refreshGraph, 3000);

    // \u2500\u2500 Animation loop \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const PARTICLE_SPEED = 0.00018;
    let lastTime = null;

    function tick(now) {
      requestAnimationFrame(tick);
      if (!lastTime) { lastTime = now; return; }
      const dt = Math.min(now - lastTime, 50);
      lastTime = now;
      particles.forEach(p => { p.phase = (p.phase + PARTICLE_SPEED * dt) % 1.0; });
      if (stabilised) network.redraw();
    }
    requestAnimationFrame(tick);

    // \u2500\u2500 Canvas: breathing halo + particles \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    network.on('afterDrawing', ctx => {
      const now     = performance.now();
      const breathe = 0.45 + 0.35 * Math.sin(now * 0.00075);

      visNodes.forEach(n => {
        if (n.hidden) return;
        const pos = network.getPosition(n.id);
        if (!pos) return;
        ctx.save();
        ctx.globalAlpha = breathe * 0.10;
        ctx.fillStyle   = n._color;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, n.size * 1.75, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      particles.forEach(p => {
        const edge = visEdges.get(p.edgeId);
        if (edge && edge.hidden) return;
        const fromPos = network.getPosition(p.fromId);
        const toPos   = network.getPosition(p.toId);
        if (!fromPos || !toPos) return;

        const t     = p.phase;
        const x     = fromPos.x + (toPos.x - fromPos.x) * t;
        const y     = fromPos.y + (toPos.y - fromPos.y) * t;
        const alpha = Math.sin(t * Math.PI) * 0.82;

        ctx.save();
        ctx.globalAlpha  = alpha;
        ctx.shadowBlur   = 7;
        ctx.shadowColor  = p.color;
        ctx.fillStyle    = p.color;
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });
    });
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP HANDLER
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress per-request noise from server logs

    def _send(self, code: int, body, content_type: str) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            p = PROJECT_ROOT / "graph2d.html"
            if p.exists():
                self._send(200, p.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(200, _HTML, "text/html; charset=utf-8")

        elif self.path == "/2d":
            self._send(200, _HTML, "text/html; charset=utf-8")

        elif self.path == "/graph":
            try:
                data = _build_graph_data()
                self._send(200, json.dumps(data, ensure_ascii=False), "application/json")
            except Exception as exc:
                log.warning("viz /graph error: %s", exc)
                self._send(500, "{}", "application/json")

        elif self.path == "/changes":
            try:
                with _gs._lock:
                    rows = _gs._conn.execute(
                        """
                        SELECT id, strftime(ts, '%H:%M:%S'), operation, category, label, detail
                        FROM graph_changes
                        ORDER BY id DESC
                        LIMIT 50
                        """
                    ).fetchall()
                changes = [
                    {"id": r[0], "ts": r[1], "operation": r[2],
                     "category": r[3], "label": r[4], "detail": r[5]}
                    for r in rows
                ]
                self._send(200, json.dumps(changes, ensure_ascii=False), "application/json")
            except Exception as exc:
                log.warning("viz /changes error: %s", exc)
                self._send(500, "[]", "application/json")

        elif self.path == "/health":
            self._send(200, b"ok", "text/plain")

        elif self.path.startswith("/lib/"):
            rel      = self.path[5:].lstrip("/")
            safe     = (PROJECT_ROOT / "lib" / rel).resolve()
            lib_root = (PROJECT_ROOT / "lib").resolve()
            if not str(safe).startswith(str(lib_root)):
                self._send(403, b"forbidden", "text/plain")
                return
            if safe.exists():
                if rel.endswith(".css"):
                    ct = "text/css"
                elif rel.endswith(".mp3"):
                    ct = "audio/mpeg"
                elif rel.endswith(".ogg"):
                    ct = "audio/ogg"
                elif rel.endswith(".wav"):
                    ct = "audio/wav"
                else:
                    ct = "application/javascript"
                self._send(200, safe.read_bytes(), ct)
            else:
                self._send(404, b"not found", "text/plain")

        elif self.path.startswith("/svg/"):
            rel      = self.path[5:].lstrip("/")
            safe     = (PROJECT_ROOT / "svg" / rel).resolve()
            svg_root = (PROJECT_ROOT / "svg").resolve()
            if not str(safe).startswith(str(svg_root)):
                self._send(403, b"forbidden", "text/plain")
                return
            if safe.exists():
                self._send(200, safe.read_bytes(), "image/svg+xml")
            else:
                self._send(404, b"not found", "text/plain")


        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, b"not found", "text/plain")
            return

        if _handle_chat_turn is None:
            self._send(503, b"chat not available", "text/plain")
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data    = json.loads(body)
            message = (data.get("message") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            self._send(400, b"bad json", "text/plain")
            return

        if not message:
            self._send(400, b"empty message", "text/plain")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        gen = _handle_chat_turn(message)
        try:
            for event in gen:
                self.wfile.write(event.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.warning("Chat stream error: %s", exc)
        finally:
            gen.close()


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def start(gs, port: int = PORT, handle_chat_turn=None) -> None:
    """
    Start the viz HTTP server in a daemon thread, then open the browser.
    gs must be a live GraphStore instance — its _lock and _conn are shared
    directly, so no second DuckDB connection is opened.
    handle_chat_turn: optional callable(message: str) → SSE event generator.
    """
    global _gs, _handle_chat_turn
    _gs               = gs
    _handle_chat_turn = handle_chat_turn

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="viz-server").start()

    # Wait up to 3 s for the server socket to accept requests
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.05)

    url = f"http://127.0.0.1:{port}"
    log.info("Graph viz: %s", url)
    webbrowser.open(url)
