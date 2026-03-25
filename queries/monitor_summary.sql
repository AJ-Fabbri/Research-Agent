-- Latest monitor run per session, showing current thesis health.
-- Usage: sqlite3 aria_research.db < queries/monitor_summary.sql

SELECT
    substr(s.id, 1, 8)          AS session_id,
    substr(s.created_at, 1, 10) AS session_date,
    s.ticker,
    s.thesis_status,
    substr(m.checked_at, 1, 10) AS last_checked,
    m.status                    AS last_status,
    substr(m.summary, 1, 100)   AS summary
FROM sessions s
JOIN (
    SELECT session_id, max(checked_at) AS latest
    FROM monitor_runs
    GROUP BY session_id
) latest_run ON s.id = latest_run.session_id
JOIN monitor_runs m
    ON m.session_id = latest_run.session_id
   AND m.checked_at = latest_run.latest
ORDER BY m.checked_at DESC;
