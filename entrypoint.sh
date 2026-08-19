#!/usr/bin/env sh
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn hpc_mcp_server_cpu.main:app --host "$HOST" --port "$PORT"
