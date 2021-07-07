PYTHON = python3

# TODO: venv setup

.PHONY = help setup train run clean

DATASETS = ATM Linkedin
NCLUSTERS = 2 3 4 5

.DEFAULT_GOAL = help

help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project type make setup"
	@echo "To train the model type make train"
	@echo "To infer clusters and compare results type make run"
	@echo "------------------------------------"


setup:

	@echo "Checking if data files are in place...";
	# booking data
	[ -d data/booking ] || (echo "Booking data is preparing" && ${PYTHON} utils/split_booking.py);
	echo "Booking data is in place";
	# synthetic data
	for C in ${NCLUSTERS}; do \
		echo "Current data directory is sin_K$${C}_C5"; \
		[ -d data/sin_K$${C}_C5 ] || (echo "No directory found, generating..." && \
		${PYTHON} generate_synthdata.py --n_clusters $${C} --n_nodes 5 --sim_type sin); \
		echo "Directory is in place"; \
	done

train:
	@echo "Training is starting...";
	${PYTHON} run.py --path_to_files data/ATM --n_clusters 1 --n_classes 7 --save_dir ATM2 --true_clusters 3 

run:
	@echo "Inference is starting...";
	for DS in ${DATASETS}; do \
		${PYTHON} cohortneyclusters.py --dataset $${DS} --experiment_n exp_0; \
		${PYTHON} tsfreshclusters.py --dataset $${DS} --experiment_n exp_0; \
	done

clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
