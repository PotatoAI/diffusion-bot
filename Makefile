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

PROJECT="lubov8"
# MODEL=/mnt/work/gnzh/dreambooth/$(PROJECT)/out
MODEL=stabilityai/stable-diffusion-2-1-base
# MODEL=CompVis/stable-diffusion-v1-4

run-model:
	env MODEL_ID=$(PROJECT) MODEL=$(MODEL) PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py

BSRGAN:
	git clone git@github.com:PotatoAI/BSRGAN.git BSRGAN

sync-outputs:
	rsync -v --size-only images/*.png /mnt/work/gnzh/dreambooth/all-outputs/

mount:
	sudo mount -t drvfs "Z:" /mnt/work
