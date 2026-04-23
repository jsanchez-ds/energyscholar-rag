#!/usr/bin/env bash
# bootstrap.sh — one-shot local setup for EnergyScholar
set -euo pipefail

echo "🚀 Bootstrapping EnergyScholar..."

if [[ ! -d .venv ]]; then
  echo "→ Creating .venv (Python 3.11)"
  py -3.11 -m venv .venv 2>/dev/null || python3.11 -m venv .venv
fi

# shellcheck disable=SC1091
if [[ -f .venv/Scripts/activate ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

pip install --upgrade pip

echo "→ Installing PyTorch CPU"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

echo "→ Installing requirements"
pip install -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "⚠️  Edit .env and paste at least one API key (Groq recommended)."
fi

mkdir -p data/papers data/cache evaluation/reports

echo ""
echo "✅ Setup complete. Next:"
echo "   source .venv/Scripts/activate    # (Windows)"
echo "   make qdrant-up"
echo "   make ingest"
echo "   make index"
echo "   make serve &"
echo "   make dashboard &"
