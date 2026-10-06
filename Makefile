.PHONY: setup data train eval demo test lint clean

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	pre-commit install || true

data:
	python scripts/download_sen1floods11.py
	python scripts/download_xbd.py

train:
	python scripts/train.py --config configs/flood.yaml

eval:
	python scripts/evaluate.py --config configs/flood.yaml

demo:
	streamlit run src/geohazard/dashboard/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
