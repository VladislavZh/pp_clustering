# COHORTNEY: Deep Clustering for Heterogeneous Event Sequences
Here we provide the code of the Cohortney method.
## Data
We provide 15 datasets: 12 synthetic and 3 real world. All the datasets are
in 'data' folder.
- data/[sin_,trunc_]Kx_C5 - synthetic datasets
- data/[Age,Linkedin,IPTV] - real world datasets
##  Method
We use LSTM based model to estimate the intensity as
a piecewise constant function. The model is in 'models/LSTM.py'.

```get_partition``` function in 'utils/data_preprocessor.py' preprocesses
point processes to the format that is suitable for LSTM

'data\trainers.py' consists the Trainer class, that conducts model training

## Starting the experiments
To start the experiments one needs to run the following command (e.g. for K5_C5
dataset):

```
python run.py --path_to_files data/K5_C5 --n_steps 128 --n_clusters 1
--true_clusters 5 --upper_bound_clusters 10 --random_walking_max_epoch 40
--n_classes 5 --lr 0.1 --lr_update_param 0.5 --lr_update_tol 25 --n_runs 5
--save_dir K5_C5 --max_epoch 50 --max_m_step_epoch 10 --min_lr 0.001
--updated_lr 0.001 --max_computing_size 800 --device cuda:0
```

All the results and the parameters are stored in 'experiments/[save_dir]' folder.

'experiments/[save_dir]/args.json' has the parameters.

'experiments/[save_dir]/last_model.pt' has the model.