"""
Created on 8/31/2020 11:19 AM

@author: Tingfeng Li, <tli@nec-labs.com>, NEC Laboratories America, Inc.
"""
import numpy as np
import torch
from itertools import combinations, permutations
from util.utils import pdist

class PairSelector:
    """
    Implementation should return indices of positive pairs and negative pairs that will be passed to compute
    Contrastive Loss
    return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.
    If balance is True, negative pairs are a random sample to match the number of positive samples
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        print('distance matrix ', distance_matrix.shape)
        print('lables ', labels.shape)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        print('all_pairs ', all_pairs.shape)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        print('positive_pairs ', positive_pairs.shape)
        print('negative_pairs ', negative_pairs.shape)

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        print('negative_distances ', negative_distances.shape)
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

class HardPositivePairSelector(PairSelector):
    """
    Creates all possible negative pairs. For positive pairs, pairs with largest distance are taken into consideration,
    matching the number of negative pairs.
    """

    def __init__(self, cpu=True):
        super(HardPositivePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        # print('distance matrix ', distance_matrix.shape)
        # print('lables ', labels.shape)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        #print('all_pairs ', all_pairs.shape)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        # print('positive_pairs ', positive_pairs.shape)
        # print('negative_pairs ', negative_pairs.shape)

        positive_distances = distance_matrix[positive_pairs[:, 0], positive_pairs[:, 1]]
        positive_distances = positive_distances.cpu().data.numpy()

        #print('negative_distances ', negative_distances.shape)
        top_positives = np.argpartition(-positive_distances, len(negative_pairs))[:len(negative_pairs)]
        top_positive_pairs = positive_pairs[torch.LongTensor(top_positives)]

        return top_positive_pairs, negative_pairs

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self, target_label=None):
        super(AllTripletSelector, self).__init__()
        self.target_label = target_label

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        if self.target_label is None:
            self.target_label = set(labels)
        for label in self.target_label:
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            #if self.target_label is None:TODO
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            #else:
            #    anchor_positives = list(permutations(label_indices, 2))

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    #return hard_negative if loss_values[hard_negative] > 0 else None
    return hard_negative


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, target_label=None, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.target_label = target_label
        self.negative_selection_fn = negative_selection_fn


    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        self.pos_dist = []
        self.neg_dist = []
        self.loss_value = []

        if self.target_label is None:
            self.target_label = set(labels)

        for label in self.target_label:
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            if self.target_label is None:#TODO
                anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            else:
                anchor_positives = list(permutations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                neg = distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), \
                                                            torch.LongTensor(negative_indices)]
                #print('ap ', ap_distance, ' neg ', torch.mean(neg))
                self.pos_dist.append(ap_distance.item())
                self.neg_dist.append(torch.mean(neg).item())

                loss_values = ap_distance - neg + self.margin
                loss_values = loss_values.data.cpu().numpy()
                self.loss_value.append(np.mean(loss_values))
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
            #print('pos ', np.mean(self.pos_dist), ' neg ', np.mean(self.neg_dist), ' loss ', np.mean(self.loss_value))

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False, target_label=None): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu, target_label=target_label)


def RandomNegativeTripletSelector(margin, cpu=False, target_label=None): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu, target_label=target_label)


def SemihardNegativeTripletSelector(margin, cpu=False, target_label=None): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu, target_label=target_label)