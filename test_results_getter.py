import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import json

param1 = 215
param2 = 235


def format_title(string):
    if len(string) < param1:
        extra = (param1 - len(string)) // 2
        string = '-' * (extra // 2) + string + '-' * (extra // 2)
        return string
    return string


def format_subtitle(string):
    if len(string) < param2:
        extra = (param2 - len(string)) // 2
        string = ' ' * (extra // 4) + '- ' * (extra // 8) + string + ' -' * (extra // 8) + ' ' * (extra // 4)
        return string
    return string


if __name__ == "__main__":
    # reading parameters
    with open("base_config.json", "r") as f:
        base_params = json.load(f)
    with open("exp_config.json", 'r') as f:
        exp_params = json.load(f)
    path_to_data = base_params['path_to_files']
    for key in exp_params.keys():
        for param_to_test in exp_params[key]:
            path_to_exp = 'experiments/' + base_params['save_dir'] + '/test_{}_{}'.format(key, param_to_test)
            exp = 'test_{}_{}'.format(key, param_to_test)
            exp_runs = os.listdir(path_to_exp)
            print(format_title('Exp = {}'.format(exp)))
            purs = []
            for run in exp_runs:
                run_path = path_to_exp + '/' + run + '/results.pkl'
                try:
                    with open(run_path, 'rb') as f:
                        print(format_subtitle('Run = {}'.format(run)))
                        res = np.array(pickle.load(f))
                        print('Best results are {}'.format(res[-1]))
                        purs.append(res[-1][1])
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5 * 15 / 15))
                        axs[0].plot(res[:, 0])
                        axs[0].set_title('loss')
                        axs[1].plot(res[:, 1])
                        axs[1].set_title('purity')
                        axs[2].plot(res[:, 2])
                        axs[2].set_title('cluster partition')
                        plt.show()
                except:
                    print('Run = {} still in progress'.format(run))
                    continue
            print('Purity for exp {}: {}+-{}'.format(exp, np.mean(purs), np.std(purs)))
            print('-' * 122)
