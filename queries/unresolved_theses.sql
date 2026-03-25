-- Active theses with no outcome recorded, oldest first (highest priority for review).
-- Usage: sqlite3 aria_research.db < queries/unresolved_theses.sql

SELECT
    substr(s.id, 1, 8)          AS id,
    substr(s.created_at, 1, 10) AS date,
    s.ticker,
    s.thesis_status             AS status,
    cast(julianday('now') - julianday(s.created_at) AS INTEGER) AS days_open,
    s.thesis,
    s.failure_conditions
FROM sessions s
LEFT JOIN outcomes o ON s.id = o.session_id
WHERE o.session_id IS NULL
  AND s.thesis IS NOT NULL
ORDER BY s.created_at ASC;
