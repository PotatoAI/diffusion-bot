POER = poetry run python
POE = poetry run

login:
	$(POE) hugginface-cli login

run:
	$(POER) src/main.py
