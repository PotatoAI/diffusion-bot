POER = poetry run python
POE = poetry run

login:
	$(POE) hugginface-cli login

run:
	env PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py

PROJECT="max1"

run-model:
	env MODEL=/mnt/work/gnzh/dreambooth/$(PROJECT)/out PYTORCH_ENABLE_MPS_FALLBACK=1 $(POER) src/main.py

BSRGAN:
	git clone git@github.com:PotatoAI/BSRGAN.git BSRGAN

sync-outputs:
	cp -v images/*.png /mnt/work/gnzh/dreambooth/all-outputs/
