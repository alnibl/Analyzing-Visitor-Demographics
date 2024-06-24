DOCKER_TAG := latest
DOCKER_IMAGE := sack_detect

.PHONY: run_app
run_app:
	python3 src\app.py

.PHONY: install
install:
	pip install -r requirements.txt


.PHONY: build
build:
	docker build -f Dockerfile . -t $(DOCKER_IMAGE):$(DOCKER_TAG)
