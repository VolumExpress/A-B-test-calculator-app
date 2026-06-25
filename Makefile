.PHONY: run test

run:
	streamlit run app.py

test:
	pytest -q
