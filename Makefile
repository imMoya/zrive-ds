reformat:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

test:
	uv run pytest tests
