BEGIN;

-- =========================================================
-- 1. Add UUID relationship columns
--    Keep existing TEXT columns temporarily for compatibility.
-- =========================================================

ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS user_uuid UUID;

ALTER TABLE messages
ADD COLUMN IF NOT EXISTS session_uuid UUID;


-- =========================================================
-- 2. Populate UUID relationships from existing identifiers
-- =========================================================

UPDATE sessions s
SET user_uuid = u.id
FROM users u
WHERE s.user_id = u.user_id
  AND s.user_uuid IS NULL;


UPDATE messages m
SET session_uuid = s.id
FROM sessions s
WHERE m.session_id = s.session_id
  AND m.session_uuid IS NULL;


-- =========================================================
-- 3. Migration safety checks
-- =========================================================

DO $$
BEGIN

    IF EXISTS (
        SELECT 1
        FROM sessions
        WHERE user_uuid IS NULL
    ) THEN
        RAISE EXCEPTION
            'Migration failed: sessions.user_uuid contains NULL values.';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM messages
        WHERE session_uuid IS NULL
    ) THEN
        RAISE EXCEPTION
            'Migration failed: messages.session_uuid contains NULL values.';
    END IF;

END $$;


-- =========================================================
-- 4. Add UUID foreign keys
-- =========================================================

ALTER TABLE sessions
ADD CONSTRAINT sessions_user_uuid_fkey
FOREIGN KEY (user_uuid)
REFERENCES users(id);


ALTER TABLE messages
ADD CONSTRAINT messages_session_uuid_fkey
FOREIGN KEY (session_uuid)
REFERENCES sessions(id);


-- =========================================================
-- 5. Enforce unique phone for current single-college V1
-- =========================================================

ALTER TABLE users
ADD CONSTRAINT users_phone_key UNIQUE (phone);


COMMIT;