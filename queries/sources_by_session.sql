-- All web sources used in a given session, grouped by search purpose.
-- Usage: sqlite3 aria_research.db "SELECT * FROM ($(cat queries/sources_by_session.sql))"
-- Or interactively:
--   sqlite3 aria_research.db
--   .param set :session_id 'your-session-id-here'
--   SELECT purpose, title, url FROM sources WHERE session_id = :session_id ORDER BY purpose;

SELECT
    purpose,
    title,
    url
FROM sources
WHERE session_id = :session_id
ORDER BY purpose, title;
