import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .encoder import TSEncoder
from .loss_func import contrastive_loss
from .loss_func import sample_contrastive_loss, observation_contrastive_loss, patient_contrastive_loss, trial_contrastive_loss
from utils import shuffle_feature_label
from utils import MyBatchSampler
from torchmetrics import MeanMetric


class COMET:
    """The COMET model"""
    
    def __init__(
        self,
        input_dims,
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
        multi_gpu=True
    ):
        """ Initialize a COMET model.
        
        Args:
            input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (str): The gpu used for training and inference.
            lr (float): The learning rate.
            batch_size (int): The batch size of samples.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
            multi_gpu (bool): A flag to indicate whether using multiple gpus
        """
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.multi_gpu = multi_gpu
        # gpu_idx_list = [0, 1]
        
        self.momentum = self.momentum
        self.queue_size = queue_size
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
        
        # create the queue
        self.register_buffer('queue', torch.randn(queue_size, length, output_dims), device=device)
        
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long), device=device)
    
    
    def fit(self, X, y, shuffle_function='trial', masks=None, factors=None, n_epochs=None, verbose=True):
        """ Training the COMET model.
        
        Args:
            X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_function (str): specify the shuffle function.
            masks (list): A list of masking functions applied (str). [Patient, Trial, Sample, Observation].
            factors (list): A list of loss factors. [Patient, Trial, Sample, Observation].
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            epoch_loss_list: a list containing the training losses on each epoch.
            epoch_f1_list: a list containing the linear evaluation on validation f1 score on each epoch.
        """
        assert X.ndim == 3
        assert y.shape[1] == 3
        # Important!!! Shuffle the training set for contrastive learning pretraining. Check details in utils.py.
        X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(torch.from_numpy(X).to(torch.float), torch.from_numpy(y).to(torch.float))
        if shuffle_function == 'random':
            train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                      drop_last=False)
        else:
            # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=min(self.batch_size, len(train_dataset)), drop_last=False)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        optimizer = torch.optim.AdamW(self.net_q.parameters(), lr=self.lr)
        
        epoch_loss_list = []
        
        # default setting
        if masks is None:
            masks = ['all_true', 'all_true', 'continuous', 'continuous']

        if factors is None:
            factors = [0.25, 0.25, 0.25, 0.25]
                    
        for epoch in range(n_epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                # count by iterations
                x = x.to(self.device)
                pid = y[:, 1]  # patient id
                tid = y[:, 2]  # trial id
                
                with torch.no_grad():
                    self._momentum_update_key_encoder()
                    
                optimizer.zero_grad()

                if factors[0] != 0:
                    # do augmentation and compute representation
                    patient_out1 = self.net_q(x, mask=masks[0])
                    patient_out2 = self._net_k(x, mask=masks[0])

                    # loss calculation
                    patient_loss = contrastive_loss(
                        patient_out1,
                        patient_out2,
                        patient_contrastive_loss,
                        id=pid,
                        hierarchical=False,
                        factor=factors[0],
                    )
                else:
                    patient_loss = 0

                if factors[1] != 0:
                    trial_out1 = self.net_q(x, mask=masks[1])
                    trial_out2 = self._net_k(x, mask=masks[1])

                    trial_loss = contrastive_loss(
                        trial_out1,
                        trial_out2,
                        trial_contrastive_loss,
                        id=tid,
                        hierarchical=False,
                        factor=factors[1],
                    )
                else:
                    trial_loss = 0

                if factors[2] != 0:
                    sample_out1 = self.net_q(x, mask=masks[2])
                    sample_out2 = self._net_k(x, mask=masks[2])

                    sample_loss = contrastive_loss(
                        sample_out1,
                        sample_out2,
                        sample_contrastive_loss,
                        hierarchical=True,
                        factor=factors[2],
                    )
                else:
                    sample_loss = 0

                if factors[3] != 0:
                    observation_out1 = self.net_q(x, mask=masks[3])
                    observation_out2 = self._net_k(x, mask=masks[3])

                    observation_loss = contrastive_loss(
                        observation_out1,
                        observation_out2,
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
           
            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if verbose:
                print(f"Epoch #{epoch+1}: loss={cum_loss}")
                
            # ranodmly select an output representation from 4 levels to enqueue
            while True:
                idx = torch.randint(0, 4, (1,))
                if factors[idx] == 0:
                    continue
                else:
                    if idx == 0:
                        self._dequeue_and_enqueue(patient_out1)
                    elif idx == 1:
                        self._dequeue_and_enqueue(trial_out1)
                    elif idx == 2:
                        self._dequeue_and_enqueue(sample_out1)
                    elif idx == 3:
                        self._dequeue_and_enqueue(observation_out1)
                    break
            
        return epoch_loss_list
    
    
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.net.parameters(), self.net_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def _net_k(self, x, mask=None):
        """ Pooling the representation.

        """
        # shuffle BN
        idx = torch.randperm(x.size(0), device=x.device)
        k = self.net_k(x[idx], mask=mask)
        k = k[torch.argsort(idx)]
        
        return k 
    
             
    def eval_with_pooling(self, x, mask=None):
        """ Pooling the representation.

        """
        out = self.net(x, mask)
        # representation shape: B x O x Co --->  B x Co
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).squeeze(-1)
        return out
    
    
    def encode(self, X, mask=None, batch_size=None):
        """ Compute representations using the model.
        
        Args:
            X (numpy.ndarray): The input data. This should have a shape of (n_samples, sample_timestamps, features).
            mask (str): The mask used by encoder can be specified with this parameter. Check masking functions in encoder.py.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        """
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
        """ Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        """ Load the model from a file.
        
        Args:
            fn (str): filename.
        """
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
    
