-- Outcome accuracy breakdown by model.
-- Usage: sqlite3 aria_research.db < queries/outcome_accuracy.sql

SELECT
    s.model_name,
    count(*)                                                        AS total,
    sum(CASE WHEN o.result = 'correct'   THEN 1 ELSE 0 END)        AS correct,
    sum(CASE WHEN o.result = 'incorrect' THEN 1 ELSE 0 END)        AS incorrect,
    sum(CASE WHEN o.result = 'partial'   THEN 1 ELSE 0 END)        AS partial,
    printf('%.0f%%',
        100.0 * sum(CASE WHEN o.result = 'correct' THEN 1 ELSE 0 END)
        / count(*))                                                  AS accuracy
FROM sessions s
INNER JOIN outcomes o ON s.id = o.session_id
GROUP BY s.model_name
ORDER BY accuracy DESC;
