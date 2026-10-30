#!/bin/bash
# Abre la aplicación local; datos y versiones permanecen en esta carpeta.
set -e
cd "$(dirname "$0")"
echo "Roya del café: preparando la aplicación local..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Se necesita Python 3.11 o posterior. Instálelo desde https://www.python.org/downloads/"
  read -r -p "Pulse Enter para cerrar. "
  exit 1
fi
if ! python3 -c 'import sys; assert sys.version_info >= (3,11)' 2>/dev/null; then
  echo "Se necesita Python 3.11 o posterior."
  read -r -p "Pulse Enter para cerrar. "
  exit 1
fi
if [ ! -f .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import streamlit, roya; assert roya.__version__ == "0.11.1"' 2>/dev/null; then
  .venv/bin/python -m pip install -e .
fi
echo "La aplicación se abrirá en el navegador. Para cerrarla, pulse Control+C aquí."
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.headless false --browser.gatherUsageStats false
