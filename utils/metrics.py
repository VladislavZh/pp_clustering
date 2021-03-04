"""
    This file contains important metrics
"""
import torch


def log_likelihood_single(partitions, lambdas, dts):
    """
        input:
               partitions - torch.Tensor, size = (batch size, sequence length, number of classes + 1), input data
               lambdas - torch.Tensor, size = (batch size, sequence length, number of classes), model output
               dts - torch.Tensor, size = (batch size), delta times for each sequence

        output:
               log likelihood - torch.Tensor, size = (1)
    """
    tmp1 = lambdas * dts[:, None, None]
    p = partitions[:, :, 1:]
    return torch.sum(tmp1) - torch.sum(p * torch.log(tmp1))


def purity(learned_ids, gt_ids):
    """
        input:
               learned_ids - torch.Tensor, labels obtained from model
               gt_ids - torch.Tensor, ground truth labels

        output:
               purity - float, purity of the model
    """
    assert len(learned_ids) == len(gt_ids)
    pur = 0
    ks = torch.unique(learned_ids)
    js = torch.unique(gt_ids)
    for k in ks:
        inters = []
        for j in js:
            inters.append(((learned_ids == k) * (gt_ids == j)).sum().item())
        pur += 1. / len(learned_ids) * max(inters)

    return pur
