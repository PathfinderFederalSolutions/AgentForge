.PHONY: build kind-load deploy logs smoke trivy bandit sbom verify deploy-verify lint-yaml codacy

IMAGE ?= agentforge:latest

build:
	docker build -t $(IMAGE) .

kind-load:
	kind load docker-image $(IMAGE) --name kind

deploy:
	kubectl apply -k k8s/staging

logs:
	kubectl logs -f deploy/nats-worker -n agentforge-staging

smoke:
	python scripts/nats_setup.py
	python scripts/smoke_publish.py

trivy:
	trivy image --severity HIGH,CRITICAL $(IMAGE) || true

sbom:
	pip install -q cyclonedx-bom && cyclonedx-py --format json --outfile sbom.json

bandit:
	pip install -q bandit && bandit -q -r swarm services || true

lint-yaml:
	bash -lc 'source source/bin/activate && pip install -q yamllint && yamllint -d "{extends: default, rules: {line-length: {max: 140}, document-start: disable}}" k8s/staging/*.yaml'

verify:
	bash scripts/verify_staging.sh

deploy-verify: deploy verify

# Optional: run Codacy static analysis if codacy-analysis-cli is installed
codacy:
	@if command -v codacy-analysis-cli >/dev/null 2>&1; then \
	  codacy-analysis-cli analyze -d . --allow-dirty --format text || true; \
	else \
	  echo "codacy-analysis-cli not installed; skipping"; \
	fi