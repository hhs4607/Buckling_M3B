#!/bin/bash
if [ -d "venv" ]; then
    echo "Using virtual environment..."
    ./venv/bin/python3 gui_buckling.py
else
    echo "Using system python..."
    python3 gui_buckling.py
fi
