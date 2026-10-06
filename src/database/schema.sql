-- Drop tables if they exist, so this script is safely re-runnable during development.
-- CASCADE also removes anything depending on these tables (like foreign key references).
DROP TABLE IF EXISTS experiment_results CASCADE;
DROP TABLE IF EXISTS user_assignments CASCADE;
DROP TABLE IF EXISTS experiments CASCADE;

CREATE TABLE experiments (
    experiment_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    primary_metric TEXT NOT NULL
);

CREATE TABLE user_assignments (
    assignment_id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id),
    user_id INTEGER NOT NULL,
    group_name TEXT NOT NULL CHECK (group_name IN ('treatment', 'control'))
);

CREATE TABLE experiment_results (
    result_id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(experiment_id),
    user_id INTEGER NOT NULL,
    converted INTEGER NOT NULL,
    revenue NUMERIC NOT NULL DEFAULT 0,
    session_duration_minutes NUMERIC
);

DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    is_new_user BOOLEAN NOT NULL,
    user_value_segment TEXT NOT NULL CHECK (user_value_segment IN ('low', 'medium', 'high')),
    pre_experiment_purchases INTEGER,
    pre_experiment_avg_session_minutes NUMERIC
);
