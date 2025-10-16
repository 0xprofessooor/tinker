#!/bin/sh
echo ---------------------
echo Installing and creating uv virtual environment
echo ---------------------
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
echo ""

echo ---------------------
echo Installing pre-commit hooks
echo ---------------------
uv run pre-commit install
echo ""

echo ---------------------
echo DONE
echo ---------------------

echo ""
echo =================================================================================
echo ""

echo ""
echo To run unit tests locally:
echo    uv run pytest

echo ""
echo =================================================================================
echo ""