-- Last 10 sessions with full thesis text and outcome.
-- Usage: sqlite3 aria_research.db < queries/recent_sessions.sql

SELECT
    substr(s.id, 1, 8)        AS id,
    substr(s.created_at, 1, 10) AS date,
    s.ticker,
    s.thesis_status           AS status,
    coalesce(o.result, 'pending') AS outcome,
    s.model_name,
    s.thesis
FROM sessions s
LEFT JOIN outcomes o ON s.id = o.session_id
ORDER BY s.created_at DESC
LIMIT 10;
