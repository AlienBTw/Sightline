.PHONY: init smoke-detector smoke-vlm test demo-api demo-ui demo-e2e docker-up train

init:
	python -m venv .venv
	.venv/Scripts/pip install -U pip
	.venv/Scripts/pip install -e ".[dev]"

smoke-detector:
	python scripts/smoke_detector.py

smoke-vlm:
	python scripts/smoke_vlm.py

test:
	python -m pytest -q

train:
	python scripts/bootstrap_labels.py --n 200
	python scripts/train_histgb.py --bootstrap data/models/bootstrap_labels.csv

demo-e2e:
	python scripts/demo_e2e_replay.py

demo-api:
	uvicorn sightline.api.main:app --reload --port 8000

demo-ui:
	streamlit run app/streamlit_app.py

docker-up:
	docker compose up --build
