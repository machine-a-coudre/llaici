#!/bin/sh
# entrypoint.sh
# ./data is bind-mounted over /app/data, hiding the dirs created in the image:
# recreate db/ here since duckdb won't create a missing parent directory.
mkdir -p /app/data/db
if [ ! -f /app/data/db/llaici.duckdb ]; then
  echo "Première initialisation..."
  duckdb -dark-mode /app/data/db/llaici.duckdb -c ".read /app/scripts/00_init.sql"
fi
exec duckdb -dark-mode -ui /app/data/db/llaici.duckdb
