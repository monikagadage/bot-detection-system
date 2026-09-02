.PHONY: help install test lint check selfcheck simulate bench serve docker

help:
	@echo "install    install dev tooling (pytest, ruff) into a venv"
	@echo "test       run the pytest suite"
	@echo "lint       run ruff"
	@echo "check      lint + tests + selfcheck (what CI runs)"
	@echo "simulate   run the end-to-end feedback-loop demo"
	@echo "bench      run all benchmarks"
	@echo "serve      run the HTTP service + dashboard on :8500"
	@echo "docker     build the container image"

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

check: lint test selfcheck

selfcheck:
	python3 selfcheck.py

simulate:
	python3 simulate.py

bench:
	@for b in benchmarks/[a-z]*.py; do echo "== $$b =="; python3 $$b; echo; done

serve:
	python3 server.py

docker:
	docker build -t botshield .
