-- Living Memory — Harlequin init
-- Loaded automatically on every session via: harlequin graph.db --init-path init.sql
INSTALL vss;
LOAD vss;
SET hnsw_enable_experimental_persistence = true;
