# -*- coding: utf-8 -*-
# @Time : 9/8/21 10:33 PM
# @Author : Tingfeng Li, <tl601@cs.rutgers.edu>, Rutgers University.
import torch

def gather_nd(params, indices, name=None):
    '''
    the input indices must be a 2d tensor in the form of [[a,b,..,c],...],
    which represents the location of the elements.
    '''
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.autograd.Variable(torch.zeros_like(indices[0]).long())
    indices = torch.autograd.Variable(indices.long())
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    return torch.take(params, idx)

def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    #seq_range = torch.range(0, max_len - 1).long()
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand)).long()

    return seq_range_expand < seq_length_expand

def _build_discounts_matrix(T, gamma):
    """Build lower-triangular matrix of discounts.
    For example for T = 3: D = [[1,       0,     0]
                               [gamma,   1,     0]
                               [gamma^2, gamma, 1]]
    Then with R, our N x T incremental rewards matrix, the discounted sum is
        R * D
    """

    def roll(x, n):
        return torch.cat((x[-n:], x[:-n]))

    power_ltri = torch.cumsum(_sequence_mask(torch.arange(0, T) + 1, T), dim=0)
    power_ltri = roll(power_ltri, 1)
    power_ltri[0] = 0

    gamma = torch.ones((T, T)) * gamma
    gamma_ltri = gamma.pow(power_ltri.float())
    gamma_ltri *= _sequence_mask(torch.arange(0, T) + 1, T).float()

    return gamma_ltri

def _get_generated_probabilities(input_batch_size, logits_seq, seq_len, coo_actions):  # TODO
    """Returns a [batch_size, seq_len] Tensor with probabilities for each
       action that was drawn
    """
    softmax_ = torch.nn.Softmax(dim=2)
    dists = softmax_(logits_seq)
    r_dists = gather_nd(dists, torch.Tensor(coo_actions).cuda())
    return r_dists.view((input_batch_size, seq_len))

def get_policy_loss(incremental_rewards, gamma, logits_seq, coo_actions):
    '''
    Input is a [batch_size, seq_len] Tensor where each entry represents
    the incremental reward for an action on a data point
    :return:
    '''
    batch_size, T = incremental_rewards.shape
    # Form matrix of discounts to apply
    gamma_ltri = _build_discounts_matrix(T, gamma).cuda()

    # Compute future discounted rewards as [batch_size x seq_len] matrix
    future_rewards = torch.mm(incremental_rewards, gamma_ltri)

    # Compute baseline and advantage
    baseline = torch.mean(future_rewards, dim=0, keepdim=True)
    advantages = future_rewards - baseline

    # Apply advantage to policy
    policy = _get_generated_probabilities(batch_size, logits_seq, T, coo_actions)

    return -(torch.log(policy) * advantages.detach()), policy[:, -1], advantages








