import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def contrastive_loss(q, k, queue, loss_func, id=None, id_queue=None,
                     hierarchical=False, factor=1.0):
    # turn off that level if factor is 0
    if factor == 0:
        return 0

    if not hierarchical:
        if id is not None:
            # pass patient and trial loss function
            return loss_func(q, k, queue, id, id_queue)
        else:
            # pass sample and observation loss function
            return loss_func(q, k)
        
    # enable hierarchical loss
    else:
        loss = torch.tensor(0., device=q.device)
        # counter for loop number
        cnt = 0
        # shorter the length of time sequence each loop
        while q.size(1) > 1:
            if id is not None:
                # pass patient and trial loss function
                loss += loss_func(q, k, queue, id, id_queue)
            else:
                # pass sample and observation loss function
                loss += loss_func(q, k)
            cnt += 1
            q = F.max_pool1d(q.transpose(1, 2), kernel_size=2).transpose(1, 2)
            k = F.max_pool1d(k.transpose(1, 2), kernel_size=2).transpose(1, 2)
            queue = F.max_pool1d(queue.transpose(1, 2), kernel_size=2).transpose(1, 2)
        return loss * factor / cnt


def sample_contrastive_loss(q, k):
    B = q.size(0)
    if B == 1:
        return q.new_tensor(0.)
    z = torch.cat([q, k], dim=0)  # 2B x O x C
    z = z.transpose(0, 1)  # O x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # O x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # O x 2B x (2B-1), left-down side, remove last zero column
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # O x 2B x (2B-1), right-up side, remove first zero column
    logits = -F.log_softmax(logits, dim=-1)  # log softmax do dividing and log
    
    i = torch.arange(B, device=q.device)
    # take all timestamps by [:,x,x]
    # logits[:, i, B + i - 1] : right-up, takes r_i and r_j
    # logits[:, B + i, i] : down-left, takes r_i_prime and r_j_prime
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def observation_contrastive_loss(q, k):
    T = q.size(1)
    if T == 1:
        return q.new_tensor(0.)
    z = torch.cat([q, k], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=q.device)
    # take all samples by [:,x,x]
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def patient_contrastive_loss(q, k, queue, pid, pid_queue):
    return id_contrastive_loss(q, k, queue, pid, pid_queue)


def trial_contrastive_loss(q, k, queue, tid, tid_queue):
    return id_contrastive_loss(q, k, queue, tid, tid_queue)


def id_contrastive_loss(q, k, queue, id, id_queue):
    id = id.cpu().detach().numpy()
    id_queue = id_queue.cpu().detach().numpy()
    
    batch_interested_matrix = np.equal.outer(id, id).astype(int) # B x B
    queue_interested_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
    interest_matrix = np.concatenate([batch_interested_matrix, queue_interested_matrix], axis=1) # B x (B + K)
    
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    B = q.size(0)
    loss = 0
    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)
    # B x O x C -> B x C x O -> B x (C x O), don't disrupt each feature
    q = q.permute(0, 2, 1).reshape((B, -1))
    k = k.permute(0, 2, 1).reshape((B, -1))
    queue = queue.permute(0, 2, 1).reshape((queue.size(0), -1))
    
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    queue = nn.functional.normalize(queue, dim=1)

    batch_sim = torch.mm(q, k.t()) # B x B
    queue_sim = torch.mm(q, queue.t()) # B x K
    sim_matrix = torch.cat([batch_sim, queue_sim], dim=1) # B x (B + K)
    temperature = 0.1
    argument = sim_matrix / temperature
    sim_matrix_exp = torch.exp(argument)

    # diag_elements = torch.diag(sim_matrix_exp)

    triu_sum = torch.sum(sim_matrix_exp, 1)  # add column
    # tril_sum = torch.sum(sim_matrix_exp, 0)  # add row

    # upper triangle same patient combs exist
    if len(rows1) > 0:
        eps = 1e-12
        triu_elements = sim_matrix_exp[rows1, cols1]  # row and column for upper triangle same patient combinations
        loss_triu = -torch.mean(torch.log((triu_elements + eps) / (triu_sum[rows1] + eps)))
        loss += loss_triu  # technically need to add 1 more term for symmetry
        return loss

    else:
        return 0
    
        
