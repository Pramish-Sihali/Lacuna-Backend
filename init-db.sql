-- Lacuna database initialization script.
-- Runs on first Docker container boot (mounted into /docker-entrypoint-initdb.d/).
-- Creates the pgvector extension, enum types, all tables, and the demo project.

-- ── Extensions ───────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Enum types ───────────────────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE claimtype AS ENUM ('supports', 'contradicts', 'extends', 'complements');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE relationshiptype AS ENUM ('prerequisite', 'builds_on', 'contradicts', 'complements', 'similar', 'parent_child');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE gaptype AS ENUM ('missing_link', 'under_explored', 'contradictory', 'isolated_concept');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ── users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              VARCHAR(255) PRIMARY KEY,
    email           VARCHAR(255) NOT NULL UNIQUE,
    name            VARCHAR(255),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

-- ── projects ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projects (
    id              SERIAL       PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    user_id         VARCHAR(255) REFERENCES users(id) ON DELETE CASCADE,
    color_index     INTEGER      DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_projects_id ON projects (id);
CREATE INDEX IF NOT EXISTS ix_projects_user_id ON projects (user_id);

-- ── documents ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL       PRIMARY KEY,
    project_id      INTEGER      NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename        VARCHAR(255) NOT NULL,
    file_path       VARCHAR(512) NOT NULL,
    file_type       VARCHAR(50)  NOT NULL,
    content_text    TEXT,
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_documents_id ON documents (id);
CREATE INDEX IF NOT EXISTS ix_documents_project_id ON documents (project_id);

-- ── chunks ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id                SERIAL       PRIMARY KEY,
    document_id       INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content           TEXT         NOT NULL,
    chunk_index       INTEGER      NOT NULL,
    embedding         vector(768),
    metadata_json     JSONB,
    extraction_status VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS ix_chunks_id ON chunks (id);
CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id);

-- ── concepts ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS concepts (
    id                  SERIAL       PRIMARY KEY,
    project_id          INTEGER      NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name                VARCHAR(255) NOT NULL,
    description         TEXT,
    generality_score    DOUBLE PRECISION,
    coverage_score      DOUBLE PRECISION,
    consensus_score     DOUBLE PRECISION,
    embedding           vector(768),
    is_gap              BOOLEAN      NOT NULL DEFAULT FALSE,
    gap_type            gaptype,
    parent_concept_id   INTEGER      REFERENCES concepts(id) ON DELETE SET NULL,
    cluster_label       INTEGER,
    metadata_json       JSONB
);
CREATE INDEX IF NOT EXISTS ix_concepts_id ON concepts (id);
CREATE INDEX IF NOT EXISTS ix_concepts_project_id ON concepts (project_id);
CREATE INDEX IF NOT EXISTS ix_concepts_name ON concepts (name);
CREATE INDEX IF NOT EXISTS ix_concepts_is_gap ON concepts (is_gap);
CREATE INDEX IF NOT EXISTS ix_concepts_parent_concept_id ON concepts (parent_concept_id);

-- ── claims ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS claims (
    id              SERIAL       PRIMARY KEY,
    document_id     INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    concept_id      INTEGER      NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    claim_text      TEXT         NOT NULL,
    claim_type      claimtype    NOT NULL,
    confidence      DOUBLE PRECISION,
    embedding       vector(768)
);
CREATE INDEX IF NOT EXISTS ix_claims_id ON claims (id);
CREATE INDEX IF NOT EXISTS ix_claims_document_id ON claims (document_id);
CREATE INDEX IF NOT EXISTS ix_claims_concept_id ON claims (concept_id);

-- ── relationships ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS relationships (
    id                  SERIAL           PRIMARY KEY,
    source_concept_id   INTEGER          NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    target_concept_id   INTEGER          NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relationship_type   relationshiptype NOT NULL,
    strength            DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    evidence_json       JSONB
);
CREATE INDEX IF NOT EXISTS ix_relationships_id ON relationships (id);
CREATE INDEX IF NOT EXISTS ix_relationships_source ON relationships (source_concept_id);
CREATE INDEX IF NOT EXISTS ix_relationships_target ON relationships (target_concept_id);

-- ── brain_state ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS brain_state (
    id              SERIAL       PRIMARY KEY,
    project_id      INTEGER      NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    last_updated    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    summary_text    TEXT,
    consensus_json  JSONB
);
CREATE INDEX IF NOT EXISTS ix_brain_state_id ON brain_state (id);
CREATE INDEX IF NOT EXISTS ix_brain_state_project_id ON brain_state (project_id);

-- ── Default demo project ────────────────────────────────────────────────────
-- Used by the unscoped (legacy) endpoints when no X-User-Id header is sent.
INSERT INTO projects (id, name, description)
VALUES (1, 'Demo Project', 'Default project for demo mode (no auth).')
ON CONFLICT (id) DO NOTHING;

-- Sync sequence so next INSERT gets id=2
SELECT setval(pg_get_serial_sequence('projects', 'id'), GREATEST((SELECT MAX(id) FROM projects), 1));
