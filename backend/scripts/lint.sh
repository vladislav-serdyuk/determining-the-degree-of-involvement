#!/bin/bash
set -e

cd "$(dirname "$0")/.."

ruff check .
mypy . --ignore-missing-imports
