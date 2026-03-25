-- Sessions older than 30 days with no outcome — candidates for cleanup or review.
-- Usage: sqlite3 aria_research.db < queries/stale_sessions.sql
-- To delete one: aria db delete <id>

SELECT
    substr(s.id, 1, 8)          AS id,
    s.id                        AS full_id,
    substr(s.created_at, 1, 10) AS date,
    s.ticker,
    cast(julianday('now') - julianday(s.created_at) AS INTEGER) AS days_old,
    s.model_name,
    substr(coalesce(s.thesis, s.query, ''), 1, 80) AS thesis_preview
FROM sessions s
LEFT JOIN outcomes o ON s.id = o.session_id
WHERE o.session_id IS NULL
  AND julianday('now') - julianday(s.created_at) > 30
ORDER BY s.created_at ASC;
