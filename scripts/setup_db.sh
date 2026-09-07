#!/usr/bin/env bash
# Create the pub17 database and enable pgvector.
#
# Docker Desktop's WSL integration is off on the dev machine, so this targets
# the native PG18 cluster on 5432. Ubuntu ships pgvector for 18 only -- the
# PG17 cluster on 5433 cannot run this project.
set -euo pipefail

PGHOST=${PGHOST:-localhost}
PGPORT=${PGPORT:-5432}
PGUSER=${PGUSER:-postgres}
export PGPASSWORD=${PGPASSWORD:-postgres}

if ! psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -tAc "SELECT 1 FROM pg_available_extensions WHERE name='vector'" | grep -q 1; then
    echo "pgvector is not installed for this server. Install it with:"
    echo "    sudo apt-get install -y postgresql-18-pgvector"
    echo "then re-run this script."
    exit 1
fi

psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -tAc "SELECT 1 FROM pg_database WHERE datname='pub17'" | grep -q 1 \
    || createdb -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" pub17
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d pub17 -c "CREATE EXTENSION IF NOT EXISTS vector"
echo "pub17 database ready on $PGHOST:$PGPORT"
