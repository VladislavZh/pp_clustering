from utils.data_preprocessor import get_dataset_for_nh
from models.CTLSTM import CTLSTMClusterwise
from utils.dataloader import CTLSTMDataset
from utils.dataloader import pad_batch_fn
from utils.likelihood_utils import generate_sim_time_seqs
from utils.trainers import TrainerClusterwiseForNH
from utils.file_system_utils import create_folder

import torch.optim as opt
from torch.utils.data import DataLoader

import torch
import pickle
import json
import numpy as np
import pandas as pd

def experiment_runner(args):
    # reading datasets
    if args['verbose']:
        print('Reading dataset')
    data, target = get_dataset_for_nh(args['path_to_files'], args['n_classes'])
    if args['verbose']:
        print('Dataset is loaded')

    # preparing folders
    if args['verbose']:
        print('Preparing folders')
    create_folder('experiments')
    path = args['save_dir'].split('/')
    for i in range(len(path)):
        create_folder('experiments/' + '/'.join(path[:i+1]))
    path_to_results = 'experiments/' + args['save_dir']

    # iterations over runs
    i = 0
    all_results = []
    while i < args['n_runs']:
        if args['verbose']:
            print('Run {}/{}'.format(i + 1, args['n_runs']))
        model = CTLSTMClusterwise(args['hidden_size'], args['n_classes'], args['n_clusters']).to(args['device'])
        optimizer = opt.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        train_dataset = CTLSTMDataset(data)
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], collate_fn=pad_batch_fn, shuffle=True)
        full_dataloader = DataLoader(train_dataset, batch_size=args['N'], collate_fn=pad_batch_fn, shuffle=False)
        best_model_path = path_to_results + '/exp_{}'.format(i) + '/best_model.pt'
        create_folder(path_to_results + '/exp_{}'.format(i))
        exp_folder = path_to_results + '/exp_{}'.format(i)
        trainer = TrainerClusterwiseForNH(model, optimizer, args['device'], args['N'], train_dataloader,
                                          full_dataloader, args['n_clusters'], target=target,
                                          max_epoch=args['max_epoch'], max_m_step_epoch=args['max_m_step_epoch'],
                                          weight_decay=args['weight_decay'], lr=args['lr'],
                                          lr_update_tol=args['lr_update_tol'],
                                          lr_update_param=args['lr_update_param'],
                                          random_walking_max_epoch=args['random_walking_max_epoch'],
                                          true_clusters=args['true_clusters'],
                                          upper_bound_clusters=args['upper_bound_clusters'],
                                          min_lr=args['min_lr'], updated_lr=args['updated_lr'],
                                          batch_size=args['batch_size'], verbose=args['verbose'],
                                          best_model_path=best_model_path,
                                          max_computing_size=None, full_purity=True)
        losses, results, cluster_part, stats = trainer.train()

        # results check
        if cluster_part is None:
            if args['verbose']:
                print('Solution failed')
            continue

        # saving results
        with open(exp_folder + '/losses.pkl', 'wb') as f:
            pickle.dump(losses, f)
        with open(exp_folder + '/results.pkl', 'wb') as f:
            pickle.dump(results, f)
        with open(exp_folder + '/stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        with open(exp_folder + '/args.json', 'w') as f:
            json.dump(args, f)
        torch.save(trainer.model, exp_folder + '/last_model.pt')
        i += 1
        results_on_run_i = np.array(results)
        print('Run = {}: {}'.format(i, results_on_run_i[-1]))
        all_results.append(results_on_run_i[np.argmin(results_on_run_i, axis=0)[0]])
    return all_results


if __name__ == "__main__":
    # reading parameters
    with open("base_config.json", "r") as f:
        base_params = json.load(f)
    with open("exp_config.json", "r") as f:
        exp_params = json.load(f)

    # iterations over experimental parameters
    best_results = []
    for key in exp_params.keys():
        params = base_params.copy()
        for param_to_test in exp_params[key]:
            if params['verbose']:
                print('Testing', key, '=', param_to_test)
            params[key] = param_to_test
            params['save_dir'] = base_params['save_dir'] + '/test_{}_{}'.format(key, param_to_test)
            print('EXP: {} = {}'.format(key, param_to_test))
            res = experiment_runner(params)
            best_results.append(res)
    for i in best_results:
        for j in i:
            print(j)
