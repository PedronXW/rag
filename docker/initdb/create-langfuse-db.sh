#!/bin/sh
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE langfuse;
    ALTER DATABASE langfuse OWNER TO "$POSTGRES_USER";
EOSQL