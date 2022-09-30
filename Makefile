POER = poetry run python
POE = poetry run

login:
	$(POE) hugginface-cli login

run:
	env PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py
