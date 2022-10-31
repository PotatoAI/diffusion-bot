POER = poetry run python
POE = poetry run

login:
	$(POE) hugginface-cli login

run:
	env PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py

.PHONY: test
test:
	$(POE) pytest

.PHONY: mypy
mypy:
	$(POE) mypy

PROJECT="max4"

run-model:
	env MODEL_ID=$(PROJECT) MODEL=/mnt/work/gnzh/dreambooth/$(PROJECT)/out PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py

BSRGAN:
	git clone git@github.com:PotatoAI/BSRGAN.git BSRGAN

sync-outputs:
	rsync -v --size-only images/*.png /mnt/work/gnzh/dreambooth/all-outputs/
