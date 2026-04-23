#!/usr/bin/env bash
# scripts/env.sh — source this before running any project command.
# (No Java/Spark needed for EnergyScholar — lighter than the flagship.)

export PATH="/c/Program Files/GitHub CLI:$PATH"
export PATH="$(pwd)/.venv/Scripts:$PATH"

echo "[env] python $(python --version 2>/dev/null)"
echo "[env] gh     $(gh --version 2>/dev/null | head -1)"
