# COHORTNEY: Deep Clustering for Heterogeneous Event Sequences
Here we provide the implementation of COHORTNEY.
The publication is currently under review.

## Data
We provide 15 datasets: 12 synthetic and 3 real-world. All datasets are
in the 'data' folder:
- data/[sin_,trunc_]Kx_C5 - synthetic datasets
- data/[Age,Linkedin,IPTV] - real world datasets

##  Method
We use an LSTM-based model to estimate the intensity as
a piecewise constant function. The model is in 'models/LSTM.py'.

### Highlights

The ```get_partition``` function in 'utils/data_preprocessor.py' preprocesses
point processes to a format that is suitable for the LSTM

The file 'data\trainers.py' consists of the Trainer class. It conducts the model training

## Starting the experiments
To start the experiments, one needs to run the following command (e.g. for K5_C5
dataset):

```
python run.py --path_to_files data/K5_C5 --n_steps 128 --n_clusters 1
--true_clusters 5 --upper_bound_clusters 10 --random_walking_max_epoch 40
--n_classes 5 --lr 0.1 --lr_update_param 0.5 --lr_update_tol 25 --n_runs 5
--save_dir K5_C5 --max_epoch 50 --max_m_step_epoch 10 --min_lr 0.001
--updated_lr 0.001 --max_computing_size 800 --device cuda:0
```

All the results and the parameters are stored in 'experiments/[save_dir]' folder:
- 'experiments/[save_dir]/args.json' has the parameters.
- 'experiments/[save_dir]/last_model.pt' has the model.
