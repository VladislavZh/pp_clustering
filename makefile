PYTHON = python3

.PHONY = help setup train run clean
.DEFAULT_GOAL = help

# available proprietary datasets
DATASETS = ATM Linkedin Age IPTV  
# default dataset for inference
INFER_DS = ATM 
# experiment numbers for inference 
NEXPERIMENTS = 0 1 2 3 4 5 6 7 8 9
# synth datasets settings
NCLUSTERS = 2 3 4 5
SYNTH = sin_K5_C5
TRUECLUSTERS = 5
SYNTHTYPE = exp


help:
	@echo "---------------HELP-----------------"
	@echo "To setup the project (download necessary datasets) type make setup"
	@echo "To train the model on all datasets type make train"
	@echo "    To train the model on synthetic dataset type make train_synthetic SYNTH=name_of_dataset TRUECLUSTERS=true_number_of_clusters"
	@echo "    To train the model on proprietary dataset type make train_name_of_dataset"
	@echo "To infer clusters and compare results type make run"
	@echo "    To run inference on specific dataset type make infer INFER_DS=name_of_dataset"
	@echo "    To run inference on booking dataset type make infer_booking"
	@echo "To obtain metrics of all datasets type make summarize_all"
	@echo "	   To obtain metrics of dataset type make summarize INFER_DS=name_of_dataset" 
	@echo "------------------------------------"
# run all
run_all: setup train run summarize_all

# download data
setup: bookingdata atmdata linkedindata syntheticdata

bookingdata: 
	@echo "Checking if booking dataset is in place...";
	@[ -d data/booking ] || (echo "Booking data is preparing" && ${PYTHON} utils/split_booking.py);
	@echo "Booking data is in place";

atmdata: 
	@echo "Checking if ATM dataset is in place...";
	@[ -d data/ATM ] || (echo "ATM data is preparing" && ${PYTHON} utils/split_atm.py);
	@echo "ATM data is in place";

linkedindata: 
	@echo "Checking if LinkedIn dataset is in place...";
	@[ -d data/Linkedin ] || (echo "LinkedIn data is preparing" && ${PYTHON} utils/split_linkedin.py);
	@echo "LinkedIn data is in place";

syntheticdata: 
	@echo "Checking if synthetic datasets are in place...";
	for T in ${SYNTHTYPE}; do \
		for C in ${NCLUSTERS}; do \
			echo "Current data directory is $${T}_K$${C}_C5"; \
			[ -d data/$${T}_K$${C}_C5 ] || (echo "No directory found, generating..." && \
			${PYTHON} utils/generate_synthdata.py --n_clusters $${C} --n_nodes 5 --sim_type $${T}  --seed 23); \
			echo "Directory is in place"; \
		done; \
	done
	
# train models
train: train_ATM train_booking train_Age train_IPTV train_Linkedin train_all_synthetic

train_ATM:
	@echo "Training on ATM dataset..."
	${PYTHON} run.py --path_to_files data/ATM --n_steps 100 --n_clusters 1 --n_classes 7 --save_dir ATM --true_clusters 7

train_booking:
	@echo "Training on Booking dataset..."
	${PYTHON} run.py --path_to_files data/booking --n_steps 12 --n_clusters 1 --n_classes 3 --save_dir booking/deviceclass --true_clusters 3 --col_to_select device_class
	${PYTHON} run.py --path_to_files data/booking --n_steps 12 --n_clusters 1 --n_classes 357 --save_dir booking/diffcheckin --true_clusters 3 --col_to_select diff_checkin
	${PYTHON} run.py --path_to_files data/booking --n_steps 12 --n_clusters 1 --n_classes 27 --save_dir booking/diffinout --true_clusters 3 --col_to_select diff_inout

train_Age:
	@echo "Training on Age dataset..."
	${PYTHON} run.py --path_to_files data/Age --n_steps 512 --n_clusters 1 --n_classes 60 --save_dir Age --true_clusters 4

train_IPTV:
	@echo "Training on IPTV dataset..."
	${PYTHON} run.py --path_to_files data/IPTV --n_steps 256 --n_clusters 1 --n_classes 16 --save_dir IPTV --true_clusters 16

train_Linkedin:
	@echo "Training on LinkedIn dataset..."
	${PYTHON} run.py --path_to_files data/Linkedin --n_steps 6 --n_clusters 1 --n_classes 82 --save_dir Linkedin --true_clusters 9

# train all synth data
train_all_synthetic:
	@echo "Training on all synthetic datasets..."
	for T in ${SYNTHTYPE}; do \
		for C in ${NCLUSTERS}; do \
			echo "Training on synthetic dataset $${T}_K$${C}_C5..."; \
			${PYTHON} run.py --path_to_files data/$${T}_K$${C}_C5 --n_steps 128 --n_clusters 1 --n_classes 5 --save_dir $${T}_K$${C}_C5 --true_clusters $${C}; \
		done; \
	done
# train specific synth dataset
train_synthetic:
	@echo "Training on synthetic dataset ${SYNTH}..."
	${PYTHON} run.py --path_to_files data/${SYNTH} --n_steps 128 --n_clusters 1 --n_classes 5 --save_dir ${SYNTH} --true_clusters ${TRUECLUSTERS}

# run inference
run: infer_all infer_booking

infer_all:
	@echo "Inference is starting...";
	# proprietary datasets
	for DS in ${DATASETS}; do \
		${PYTHON} tsfreshclusters.py --dataset $${DS}; \
		for E in ${NEXPERIMENTS}; do \
			echo "Running inference: $${DS} dataset, experiment number is $${E}"; \
			${PYTHON} cohortneyclusters.py --dataset $${DS} --experiment_n exp_$${E}; \
		done; \
	done
	# synthetic datasets
	for T in ${SYNTHTYPE}; do \ 
		for C in ${NCLUSTERS}; do \
			${PYTHON} tsfreshclusters.py --dataset $${T}_K$${C}_C5; \
			for E in ${NEXPERIMENTS}; do \
				echo "Running inference: dataset $${T}_K$${C}_C5, experiment number is $${E}"; \
				${PYTHON} cohortneyclusters.py --dataset $${T}_K$${C}_C5 --experiment_n exp_$${E}; \
			done; \
		done; \
	done

infer_booking:
	# booking dataset
	${PYTHON} tsfreshclusters.py --dataset booking --experiment_n  --col_to_select device_class; \
	${PYTHON} tsfreshclusters.py --dataset booking --experiment_n  --col_to_select diff_checkin; \ 
	${PYTHON} tsfreshclusters.py --dataset booking --experiment_n  --col_to_select diff_inout; \
	for E in ${NEXPERIMENTS}; do \
		echo "Running inference: Booking dataset, experiment number is $${E}"; \
		${PYTHON} cohortneyclusters.py --dataset booking --experiment_n deviceclass/exp_$${E} --col_to_select device_class; \ 
		${PYTHON} cohortneyclusters.py --dataset booking --experiment_n diffcheckin/exp_$${E} --col_to_select diff_checkin; \ 
		${PYTHON} cohortneyclusters.py --dataset booking --experiment_n diffinout/exp_$${E} --col_to_select diff_inout; \ 
		${PYTHON} utils/vote_booking.py; \
	done

infer: 
	# run inference on specific dataset
	${PYTHON} tsfreshclusters.py --dataset ${INFER_DS} ; \
	for E in ${NEXPERIMENTS}; do \
		echo "Running inference: dataset ${INFER_DS}, experiment number is $${E}"; \
		${PYTHON} cohortneyclusters.py --dataset ${INFER_DS} --experiment_n exp_$${E}; \
	done

# obtain statistics
summarize:
	@echo "Printing metrics' scores, dataset ${INFER_DS}"
	${PYTHON} utils/calc_purity.py --dataset ${INFER_DS}
summarize_all:
	# proprietary datasets
	for DS in ${DATASETS}; do \
		echo "Printing metrics' scores, dataset $${DS}"; \
		${PYTHON} utils/calc_purity.py --dataset $${DS}; \
	done
	# synthetic datasets
	for T in ${SYNTHTYPE}; do \
		for C in ${NCLUSTERS}; do \
			echo "Printing metrics' scores, dataset $${T}_K$${C}_C5"; \
			${PYTHON} utils/calc_purity.py --dataset $${T}_K$${C}_C5; \
		done; \
	done

# clean tmp files
clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
