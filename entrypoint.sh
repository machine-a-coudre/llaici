#!/bin/sh
# entrypoint.sh
if [ ! -f /app/data/db/llaici.duckdb ]; then
  echo "Première initialisation..."
  duckdb /app/data/db/llaici.duckdb -c ".read /app/scripts/00_init.sql"
fi
exec duckdb -ui /app/data/db/llaici.duckdb