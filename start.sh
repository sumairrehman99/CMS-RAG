#!/bin/sh

uvicorn implementation.main:app --host 0.0.0.0 --port 8000 &
python implementation/frontend.py