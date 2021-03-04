import os
import pandas as pd
import torch


def cmp_to_key(mycmp):
    """
        Convert a cmp= function into a key= function
    """

    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def compare(a, b):
    tmp1 = int(a[:-4])
    tmp2 = int(b[:-4])
    return tmp1 - tmp2


def get_partition(df, num_of_steps, num_of_classes, end_time=None):
    """
        Transforms dataset into partition

        inputs:
                df - pandas.DataFrame, columns - time and event
                num_of_steps - int, number of steps in partition
                num_of_classes - int, number of event types
                end_time - float, end time or None

        outputs:
                partition - torch.Tensor, size = (num_of_steps, num_of_classes + 1)
    """
    # setting end time if None
    if end_time is None:
        end_time = df['time'][len(df['time']) - 1]

    # preparing output template
    res = torch.zeros(num_of_steps, num_of_classes + 1)

    # finding delta time
    dt = end_time / num_of_steps
    res[:, 0] = end_time / num_of_steps

    # iterations over event sequence
    for i in range(len(df['time'])):
        k = int(df['time'][i] / dt)
        if k == num_of_steps:
            k -= 1
        res[k, int(df['event'][i]) + 1] += 1

    return res


def get_dataset(path_to_files, n_classes, n_steps):
    """
        Reads dataset

        inputs:
                path_to_files - str, path to csv files with dataset
                n_classes - int, number of event types
                n_steps - int, number of steps in partitions

        outputs:
                data - torch.Tensor, size = (N, n_steps, n_classes + 1), dataset
                target - torch.Tensor, size = (N), true labels or None
    """
    # searching for files
    files = os.listdir(path_to_files)
    target = None

    # reading target
    if 'clusters.csv' in files:
        files.remove('clusters.csv')
        target = torch.Tensor(pd.read_csv(path_to_files + '/clusters.csv')['cluster_id'])

    # reading data
    files = sorted(files, key=cmp_to_key(compare))
    data = torch.zeros(len(files), n_steps, n_classes + 1)
    for i, f in enumerate(files):
        df = pd.read_csv(path_to_files + '/' + f)
        data[i, :, :] = get_partition(df, n_steps, n_classes)

    return data, target
