import numpy as np
import torch
from torch import nn
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, euclidean

class Hausdorff_Loss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()
        
    def hausdorff_distance(self, X, Y):
        hd_xy = max([min([torch.norm(x-y) for y in Y]) for x in X])
        hd_yx = max([min([torch.norm(y-x) for x in X]) for y in Y])
        HD = max(hd_xy, hd_yx)
        return HD

class AHD_Loss_Mine(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def forward(self, X, Y):
        dis_1 = (sum([min([torch.norm(x-y) for y in Y]) for x in X])) / len(X)
        dis_2 = (sum([min([torch.norm(y-x) for x in X]) for y in Y])) / len(Y)
        loss = dis_1 + dis_2
        return loss

class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
            i.e. dist[i,j] = ||x[i,:]-y[j,:]||
        """

        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = self.cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res

