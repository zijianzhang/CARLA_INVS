SHELL:=/bin/bash

MODEL_FILE := model1.pth
CONFIG_FILE := cfgs/train2_new.yaml
# CONFIG_FILE:=cfgs/second.yaml
# CONFIG_FILE := cfgs/carla.yaml
# CONFIG_FILE:=cfgs/bear1.yaml
BATCH_SIZE:=1

all:install

install:install-spconv
	sudo apt install python3 python3-pip -y
	pip3 install -r requirements.txt --user
	python3 setup.py develop --user

install-spconv:
	@echo -e '\nPlease ensure you have CUDA 9.0+ and cuDNN installed! (press ENTER to continue...)'; read -p ''

	sudo apt install libboost-all-dev -y
	# sudo apt install llvm-10.0-dev -y; sudo ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config
	pip3 install cmake --user --upgrade
	pip3 install "torch>=1.1,<=1.4" --user
	
	rm -rf build/spconv && git clone https://github.com/traveller59/spconv.git build/spconv --recursive
	cd build/spconv; python3 setup.py bdist_wheel; cd ./dist; pip3 install *.whl --user
	#git reset --hard 8da6f967fb9a054d8870c3515b1b44eca2103634;

prepare:
	cd pcdet/datasets/kitti; python3 kitti_dataset_download.py create_kitti_infos

remove-spconv:
	pip3 uninstall spconv

test:

	# python3 test.py --cfg_file $(CONFIG_FILE) --batch_size 1 --ckpt $(MODEL_FILE) \
    # --set MODEL.RPN.BACKBONE.NAME UNetV0 MODEL.RPN.RPN_HEAD.ARGS use_binary_dir_classifier:True
	cd tools; \
	python3 test.py --cfg_file $(CONFIG_FILE) --batch_size 1 --ckpt $(MODEL_FILE)

train:
	cd tools; \
	python3 train.py --cfg_file $(CONFIG_FILE) --batch_size $(BATCH_SIZE) --workers 0 --epochs 50
	#--ckpt 

clean:
	rm -rf output
	rm -f model/checkpoint_epoch_*
