# Living Memory — Mission Brief

## What this project is

A local knowledge graph that grows through voice chat. Every conversation
automatically extracts entities and relationships stored in DuckDB.

## Stack

- LLM: Qwen3-30B-A3B via Ollama (localhost:11434)
- STT: Whisper.cpp (Metal GPU, base.en model)
- TTS: Kokoro ONNX (CPU, af_bella voice)
- Storage: DuckDB (graph.db — single local file)
- Language: Python 3.12, venv at ./venv

## Architecture priorities

1. Fully offline — no cloud dependencies
2. Tool calling for graph ops: upsert_node, add_edge, query_neighbors
3. Deduplication before every insert
4. Controlled relationship vocabulary (~15 types max)
5. Adjacency cache table refreshed on every write
6. Vector search via DuckDB VSS extension (HNSW index)

## Do not

- Install system packages with sudo
- Use pip outside the venv
- Hardcode credentials — use .env
