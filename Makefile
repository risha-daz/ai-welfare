.PHONY: smoke-test

smoke-test: 
	python -m src.run --experiment pilot --model all --limit 3 --run-id smoke