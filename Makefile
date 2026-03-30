.PHONY: install install-x86 install-dai clean

install:
	bash setup.sh

install-x86:
	bash setup.sh x86

install-dai:
	bash setup.sh dai

clean:
	find ./src -type d -name "__pycache__" -exec rm -rf {} +
	find ./src -type f -name "*.pyc" -exec rm -f {} +
	find ./src -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "pymp*" -exec rm -rf {} +
	find . -type d -name "tmp*" -exec rm -rf {} +