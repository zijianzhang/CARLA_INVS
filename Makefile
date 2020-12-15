all:dependency

dependency:
	pip3 install -r requirement.txt
	sudo apt install libxerces-c-3.2 libjpeg8

clean:
	rm -rf raw_data/ log/