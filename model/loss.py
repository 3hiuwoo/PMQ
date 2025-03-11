import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def id_momentum_contrastive_loss(q, k, queue, id, id_queue, diag=True):
    id = id.cpu().detach().numpy()
    id_queue = id_queue.cpu().detach().numpy()
    batch_interest_matrix = np.equal.outer(id, id).astype(int) # B x B
    queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
    interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    temperature = 0.1
    eps = 1e-12
    batch_sim_matrix = torch.mm(q, k.t()) # B x B
    queue_sim_matrix = torch.mm(q, queue.t()) # B x K
    sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
    argument = sim_matrix / temperature
    sim_matrix_exp = torch.exp(argument)
    
    triu_elements = sim_matrix_exp[rows1,cols1]
    loss = -torch.mean(torch.log((triu_elements+eps)/(torch.sum(sim_matrix_exp,1)[rows1]+eps)))
    
    if diag:
        diag_elements = torch.diag(sim_matrix_exp)
        loss_diag = -torch.mean(torch.log((diag_elements+eps)/(torch.sum(sim_matrix_exp,1)+eps)))
        loss += loss_diag
        loss /= 2

    return loss


def id_momentum_loss(q, k, queue, id, id_queue):
    ''' Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        q (torch.Tensor): query embeddings from model for different perturbations of same instance (NxBxH)
        k (torch.Tensor): key embeddings from model for different perturbations of same instance (NxBxH)
        queue (torch.Tensor): queue embeddings from model for different perturbations of same instance (NxBxH)
        id (list): ids of instances in batch
        id_queue (torch.Tensor): queue ids
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    '''
    id = id.cpu().detach().numpy()
    id_queue = id_queue.cpu().detach().numpy()
    batch_interest_matrix = np.equal.outer(id, id).astype(int) # B x B
    queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
    interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    temperature = 0.1
    eps = 1e-12
    batch_sim_matrix = torch.mm(q, k.t()) # B x B
    queue_sim_matrix = torch.mm(q, queue.t()) # B x K
    sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
    argument = sim_matrix / temperature
    sim_matrix_exp = torch.exp(argument)
    
    diag_elements = torch.diag(sim_matrix_exp)
    triu_elements = sim_matrix_exp[rows1,cols1]
    
    loss_diag = -torch.mean(torch.log((diag_elements+eps)/(torch.sum(sim_matrix_exp,1)+eps)))
    loss_triu = -torch.mean(torch.log((triu_elements+eps)/(torch.sum(sim_matrix_exp,1)[rows1]+eps)))
    
    loss = loss_diag + loss_triu
    loss /= 2

    return loss


def id_momentum_loss2(q, k, queue, id, id_queue):
    ''' Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        q (torch.Tensor): query embeddings from model for different perturbations of same instance (NxBxH)
        k (torch.Tensor): key embeddings from model for different perturbations of same instance (NxBxH)
        queue (torch.Tensor): queue embeddings from model for different perturbations of same instance (NxBxH)
        id (list): ids of instances in batch
        id_queue (torch.Tensor): queue ids
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    '''
    id = id.cpu().detach().numpy()
    id_queue = id_queue.cpu().detach().numpy()
    batch_interest_matrix = np.equal.outer(id, id).astype(int) # B x B
    queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
    interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)
    # only consider upper diagnoal where the queue is taken into account
    rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
    # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs

    temperature = 0.1
    eps = 1e-12
    batch_sim_matrix = torch.mm(q, k.t()) # B x B
    queue_sim_matrix = torch.mm(q, queue.t()) # B x K
    sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
    argument = sim_matrix / temperature
    sim_matrix_exp = torch.exp(argument)
    
    # diag_elements = torch.diag(sim_matrix_exp)
    triu_elements = sim_matrix_exp[rows1,cols1]
    
    # loss_diag = -torch.mean(torch.log((diag_elements+eps)/(torch.sum(sim_matrix_exp,1)+eps)))
    loss_triu = -torch.mean(torch.log((triu_elements+eps)/(torch.sum(sim_matrix_exp,1)[rows1]+eps)))
    
    # loss = loss_diag + loss_triu
    # loss /= 2

    return loss_triu # loss


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
        loss += loss_triu  # technicalneed to add 1 more term for symmetry
        return loss

    else:
        return 0
    
    

    
        
