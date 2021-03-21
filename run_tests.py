from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
from models.LSTM import LSTMMultiplePointProcesses
from utils.file_system_utils import create_folder
import torch
import pickle
import json
import numpy as np


def experiment_runner(args):
    # reading datasets
    if args['verbose']:
        print('Reading dataset')
    data, target = get_dataset(args['path_to_files'], args['n_classes'], args['n_steps'])
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
        model = LSTMMultiplePointProcesses(args['n_classes'] + 1, args['hidden_size'], args['num_layers'],
                                           args['n_classes'], args['n_clusters'], args['n_steps'],
                                           dropout=args['dropout']).to(args['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args['pretraining_lr'], weight_decay=args['weight_decay'])
        best_model_path = path_to_results + '/exp_{}'.format(i) + '/best_model.pt'
        create_folder(path_to_results + '/exp_{}'.format(i))
        exp_folder = path_to_results + '/exp_{}'.format(i)
        trainer = TrainerClusterwise(model, optimizer, args['device'], data, args['n_clusters'], target=target,
                                     alpha=args['alpha'], beta=args['beta'], epsilon=args['epsilon'],
                                     sigma_0=args['sigma_0'], sigma_inf=args['sigma_inf'], inf_epoch=args['inf_epoch'],
                                     max_epoch=args['max_epoch'], max_m_step_epoch=args['max_m_step_epoch'],
                                     max_m_step_epoch_add=args['max_m_step_epoch_add'],
                                     lr = args['lr'],
                                     lr_update_tol=args['lr_update_tol'], lr_update_param=args['lr_update_param'],
                                     lr_update_param_changer=args['lr_update_param_changer'],
                                     lr_update_param_second_changer=args['lr_update_param_second_changer'],
                                     min_lr=args['min_lr'], updated_lr=args['updated_lr'],
                                     batch_size=args['batch_size'], verbose=args['verbose'],
                                     best_model_path=best_model_path if args['save_best_model'] else None,
                                     max_computing_size=args['max_computing_size'], full_purity=args['full_purity'],
                                     pretrain_number_of_epochs=args["pretrain_number_of_epochs"],
                                     pretrain_step=args['pretrain_step'], pretrain_mul=args['pretrain_mul'],
                                     pretraining=args["pretraining"])
        losses, results, cluster_part, stats = trainer.train()

        # results check
        if cluster_part is None:
            if args['verbose']:
                print('Solution failed')
        if args['degenerate_eps']:
            if cluster_part < args['degenerate_eps'] / args['n_clusters']:
                if args['verbose']:
                    print("Degenerate solution")
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
        print('Run = {}: {}'.format(i, results_on_run_i[np.argmin(results_on_run_i, axis=0)[0]]))
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
