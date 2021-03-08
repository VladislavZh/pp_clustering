"""
    Main program
"""

from argparse import ArgumentParser
from utils.data_preprocessor import get_dataset
from utils.trainers import TrainerClusterwise
from models.LSTM import LSTMMultiplePointProcesses
from utils.file_system_utils import create_folder
import torch
import pickle
import json


def parse_arguments():
    """
        Processes and returns cmd arguments

        inputs:
                None

        outputs:
                args
    """
    parser = ArgumentParser()
    parser.add_argument('--path_to_files', type=str, required=True, help='path to data, required')
    parser.add_argument('--n_steps', type=int, default=128, help='number of steps in partitions, default - 128')
    parser.add_argument('--n_clusters', type=int, required=True, help='number of clusters, required')
    parser.add_argument('--n_classes', type=int, required=True, help='number of types of events, required')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size, default - 128')
    parser.add_argument('--num_layers', type=int, default=3, help='number of LSTM layers, default - 3')
    parser.add_argument('--dropout', type=float, default=0.3, help='LSTM dropout rate, default - 0.3')
    parser.add_argument('--lr', type=float, default=0.1, help='optimizer initial learning rate, default - 0.1')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay, default - 1e-5')
    parser.add_argument('--degenerate_eps', type=float, help='if provided, adds degeneration test with eps/n_clusters '
                                                             'and skips degenerate solutions')
    parser.add_argument('--n_runs', type=int, default=5, help='number of starts, default - 5')
    parser.add_argument('--save_dir', type=str, required=True, help='saves results to experiments/save_dir, required')
    parser.add_argument('--save_best_model', type=bool, default=True, help='if True, saves the state of the best '
                                                                           'model according to loss')
    parser.add_argument('--alpha', type=float, default=1.001, help='is used for prior distribution of lambdas, '
                                                                   'punishes small lambdas, default - 1.001')
    parser.add_argument('--beta', type=float, default=0.0005, help='is used for prior distribution of lambdas, '
                                                                   'punishes big lambdas, default - 0.0005')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='is used for log-s regularization log(x) -> log(x '
                                                                    '+ epsilon), default - 1e-8')
    parser.add_argument('--sigma_0', type=float, default=1.0, help='initial sigma of gaussian that is used for '
                                                                   'convolution with gamma for stabilization, default '
                                                                   '- 1.0')
    parser.add_argument('--sigma_inf', type=float, default=0.005, help='sigma on epoch inf_epoch, is used for '
                                                                       'computing decay, default - 0.005')
    parser.add_argument('--inf_epoch', type=int, default=50, help='when sigma_inf is achieved, used for computing '
                                                                  'decay, default - 50')
    parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs of EM algorithm, default - 50')
    parser.add_argument('--max_m_step_epoch', type=int, default=50, help='number of epochs of neural net training on '
                                                                         'M-step, default - 50')
    parser.add_argument('--lr_update_tol', type=int, default=35, help='tolerance before updating learning rate, '
                                                                      'default - 25')
    parser.add_argument('--lr_update_param', type=float, default=0.8, help='learning rate multiplier, default - 0.85')
    parser.add_argument('--lr_update_param_changer', type=float, default=1.02, help='lr_update_param multiplier, '
                                                                                    'default - 1.02')
    parser.add_argument('--batch_size', type=int, default=150, help='batch size during neural net training, default - '
                                                                    '150')
    parser.add_argument('--verbose', type=bool, default=True, help='if true, prints logs, default - True')
    parser.add_argument('--device', type=str, default='cpu', help='device that should be used for training, default - '
                                                                  'cpu')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()

    # reading datasets
    if args.verbose:
        print('Reading dataset')
    data, target = get_dataset(args.path_to_files, args.n_classes, args.n_steps)
    if args.verbose:
        print('Dataset is loaded')

    # preparing folders
    if args.verbose:
        print('Preparing folders')
    create_folder('experiments')
    create_folder('experiments/' + args.save_dir)
    path_to_results = 'experiments/' + args.save_dir

    # iterations over runs
    i = 0
    while i < args.n_runs:
        if args.verbose:
            print('Run {}/{}'.format(i + 1, args.n_runs))
        model = LSTMMultiplePointProcesses(args.n_classes + 1, args.hidden_size, args.num_layers, args.n_classes,
                                           args.n_clusters, args.n_steps, dropout=args.dropout).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_model_path = path_to_results + '/exp_{}'.format(i) + '/best_model.pt'
        create_folder(path_to_results + '/exp_{}'.format(i))
        exp_folder = path_to_results + '/exp_{}'.format(i)
        trainer = TrainerClusterwise(model, optimizer, args.device, data, args.n_clusters, target=target,
                                     alpha=args.alpha, beta=args.beta, epsilon=args.epsilon, sigma_0=args.sigma_0,
                                     sigma_inf=args.sigma_inf, inf_epoch=args.inf_epoch, max_epoch=args.max_epoch,
                                     max_m_step_epoch=args.max_m_step_epoch, lr_update_tol=args.lr_update_tol,
                                     lr_update_param=args.lr_update_param,
                                     lr_update_param_changer=args.lr_update_param_changer, batch_size=args.batch_size,
                                     verbose=args.verbose,
                                     best_model_path=best_model_path if args.save_best_model else None)
        losses, results, cluster_part, stats = trainer.train()

        # results check
        if cluster_part is None:
            if args.verbose:
                print('Solution failed')
            continue
        if args.degenerate_eps:
            if cluster_part < args.degenerate_eps / args.n_clusters:
                if args.verbose:
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
            json.dump(vars(args), f)
        torch.save(model.state_dict(), exp_folder + '/last_model.pt')
        i += 1
