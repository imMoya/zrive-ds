reformat:
	uv run ruff format src tests

lint:
	uv run ruff check --fix src tests

test:
	uv run pytest tests
