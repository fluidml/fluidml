test:
	TERM=unknown pytest --cov-report term-missing --cov=fluidml tests/ -vv
test-no-cov:
	TERM=unknown pytest tests/ -vv
format-check:
	black --check .
format:
	black .
.PHONY: docs
docs:
	cd docs; make html
