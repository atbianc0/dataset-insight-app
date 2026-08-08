.PHONY: install install-dev run lint test coverage audit check docker-build docker-run

install:
	python -m pip install --requirement requirements.txt

install-dev:
	python -m pip install --requirement requirements-dev.txt

run:
	streamlit run app.py

test:
	python -m pytest -W error

lint:
	python -m ruff check app.py src tests

coverage:
	python -m pytest --cov=src --cov-branch --cov-report=term-missing

audit:
	python -m pip_audit --requirement requirements-dev.txt

check: lint coverage audit

docker-build:
	docker build --tag datalens:latest .

docker-run:
	docker run --rm --publish 8501:8501 --env OPENAI_API_KEY datalens:latest
