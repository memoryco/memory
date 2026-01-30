#!/bin/bash
sqlite3 "$HOME/Library/Application Support/memory/brain.db" "SELECT id, LENGTH(embedding) as emb_len FROM engrams LIMIT 5;"
