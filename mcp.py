''' Referenced from MoCo official & unofficial implementations and COMET implementation.
    See:
        https://github.com/DL4mHealth/COMET
        https://github.com/facebookresearch/moco
        https://github.com/leftthomas/MoCo
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder
from model.cl_loss import contrastive_loss
from model.cl_loss import sample_contrastive_loss, observation_contrastive_loss, patient_contrastive_loss, trial_contrastive_loss
from utils import shuffle_feature_label
from utils import MyBatchSampler


class MCP:
    '''The MCP model.
    
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The hidden dimension of the encoder.
        length (int): The length of the representation/series.
        depth (int): The number of hidden residual blocks in the encoder.
        device (str): The gpu used for training and inference.
        lr (float): The learning rate.
        batch_size (int): The batch size of samples.
        momentum (float): The momentum used for the key encoder.
        queue_size (int): The size of the queue.
        num_queue (int): The number of queues.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
        callback_func (Union[Callable, NoneType]): A callback function that would be called after each epoch.
    '''
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        length=300,
        depth=10,
        device='cuda',
        lr=1e-4,
        batch_size=256,
        momentum=0.999,
        queue_size=16384,
        num_queue=1,
        multi_gpu=True,
        callback_func=None
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.multi_gpu = multi_gpu
        # gpu_idx_list = [0, 1]
        self.callback_func = callback_func
        
        self.momentum = momentum
        self.queue_size = queue_size
        # todo: multiple queues
        self.num_queue = num_queue
        
        self.net_q = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net_k = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        
        for param_q, param_k in zip(
            self.net_q.parameters(), self.net_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self.net_q = nn.DataParallel(self.net_q)
            self.net_k = nn.DataParallel(self.net_k)
        self.net_q.to(device)
        self.net_k.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net = torch.optim.swa_utils.AveragedModel(self.net_q)
        self.net.update_parameters(self.net_q)

        # projection head append after encoder
        # self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)
        
        # create the queues
        # self.register_buffer('queue', torch.randn(queue_size, length, output_dims))
        # self.register_buffer('pid_queue', torch.zeros(queue_size, dtype=torch.long))
        # self.register_buffer('tid_queue', torch.zeros(queue_size, dtype=torch.long))
        # self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.queue = torch.randn(queue_size, length, output_dims, device=device, requires_grad=False)
        self.queue = nn.functional.normalize(self.queue, dim=1)
        
        self.pid_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.tid_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        

    def fit(self, X, y, shuffle_function='trial', masks=None, factors=None, epochs=None, verbose=True):
        ''' Training the MCP model.
        
        Args:
            X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_func (str): specify the shuffle function.
            masks (list): A list of masking functions applied (str). [Patient, Trial, Sample, Observation].
            factors (list): A list of loss factors. [Patient, Trial, Sample, Observation].
            epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            epoch_loss_list: a list containing the training losses on each epoch.
        '''
        assert X.ndim == 3
        assert y.shape[1] == 3
        # Shuffle the training set for contrastive learning pretraining.
        X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == 'random':
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        optimizer = torch.optim.AdamW(self.net_q.parameters(), lr=self.lr)
        
        epoch_loss_list = []
        
        # default setting
        if masks is None:
            masks = ['all_true', 'all_true', 'continuous', 'continuous']

        if factors is None:
            factors = [0.25, 0.25, 0.25, 0.25]
        
        start_time = datetime.now()           
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                # count by iterations
                x = x.to(self.device)
                pid = y[:, 1]  # patient id
                tid = y[:, 2]  # trial id
                
                with torch.no_grad():
                    self._momentum_update_key_encoder()
                    queue = self.queue.clone().detach()
                    
                optimizer.zero_grad()
                
                if factors[0] != 0:
                    # do augmentation and compute representation
                    patient_out1 = self.net_q(x, mask=masks[0])
                    # patient_out2 = self.net_q(x, mask=masks[0])
                    patient_out2 = self._net_k(x, mask=masks[0])

                    # loss calculation
                    patient_loss = contrastive_loss(
                        patient_out1,
                        patient_out2,
                        queue,
                        patient_contrastive_loss,
                        id=pid,
                        id_queue=self.pid_queue.clone().detach(),
                        hierarchical=False,
                        factor=factors[0],
                    )
                else:
                    patient_loss = 0

                if factors[1] != 0:
                    trial_out1 = self.net_q(x, mask=masks[1])
                    # trial_out2 = self.net_q(x, mask=masks[1])
                    trial_out2 = self._net_k(x, mask=masks[1])

                    trial_loss = contrastive_loss(
                        trial_out1,
                        trial_out2,
                        queue,
                        trial_contrastive_loss,
                        id=tid,
                        id_queue=self.tid_queue.clone().detach(),
                        hierarchical=False,
                        factor=factors[1],
                    )
                else:
                    trial_loss = 0

                if factors[2] != 0:
                    sample_out1 = self.net_q(x, mask=masks[2])
                    sample_out2 = self.net_q(x, mask=masks[2])
                    # sample_out2 = self._net_k(x, mask=masks[2])

                    sample_loss = contrastive_loss(
                        sample_out1,
                        sample_out2,
                        queue,
                        sample_contrastive_loss,
                        hierarchical=True,
                        factor=factors[2],
                    )
                else:
                    sample_loss = 0

                if factors[3] != 0:
                    observation_out1 = self.net_q(x, mask=masks[3])
                    observation_out2 = self.net_q(x, mask=masks[3])
                    # observation_out2 = self._net_k(x, mask=masks[3])

                    observation_loss = contrastive_loss(
                        observation_out1,
                        observation_out2,
                        queue,
                        observation_contrastive_loss,
                        hierarchical=True,
                        factor=factors[3],
                    )
                else:
                    observation_loss = 0

                loss = patient_loss + trial_loss + sample_loss + observation_loss

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.net_q)

                cum_loss += loss.item()
                
                self._dequeue_and_enqueue(patient_out2, pid, tid)
           
            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if verbose:
                print(f"=> Epoch {epoch+1}: loss: {cum_loss}")
                
            if self.callback_func is not None:
                self.callback_func(self, epoch)
                
        end_time = datetime.now()
        print(f'Training finished in {end_time - start_time}')
            
        return epoch_loss_list
    
    
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.net.parameters(), self.net_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    def _dequeue_and_enqueue(self, keys, pid, tid):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = keys
        self.pid_queue[ptr : ptr + batch_size] = pid
        self.tid_queue[ptr : ptr + batch_size] = tid
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def _net_k(self, x, mask=None):
        '''
        perform shuffle BN before momentum encoder forward propagation
        '''
        # shuffle BN
        idx = torch.randperm(x.size(0), device=x.device)
        k = self.net_k(x[idx], mask=mask)
        k = k[torch.argsort(idx)]
        
        return k 
    
             
    def eval_with_pooling(self, x, mask=None):
        '''
        max pool the representation
        '''
        out = self.net(x, mask)
        # representation shape: B x O x Co --->  B x Co
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).squeeze(-1)
        return out
    
    
    def encode(self, X, mask=None, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            X (numpy.ndarray): The input data. This should have a shape of (n_samples, sample_timestamps, features).
            mask (str): The mask used by encoder can be specified with this parameter. Check masking functions in encoder.py.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert X.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        # n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(X).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0].to(self.device)
                # print(next(self.net.parameters()).device)
                # print(x.device)
                out = self.eval_with_pooling(x, mask)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        # return output.numpy()
        return output.cpu().numpy()


    def save(self, fn):
        '''Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        '''Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
    
