''' Train MCP
'''
import os
import argparse
import torch
import sklearn
import numpy as np
import torch.nn.functional as F
from mcp import MCP
from data import load_data
from utils import seed_everything, get_device
from eval_protocols import fit_lr


parser = argparse.ArgumentParser(description='MCP training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# for the data
parser.add_argument('--root', type=str, default='dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='chapman', help='select pretraining dataset')
parser.add_argument('--length', type=int, default=300, help='length of each sample')
# for the model
parser.add_argument('--depth', type=int, default=10, help='depth of the encoder')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the model')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum for the model')
parser.add_argument('--queue_size', type=int, default=4096, help='queue size for the model')
parser.add_argument('--num_queues', type=int, default=1, help='number of queues for the model')
parser.add_argument('--masks', type=str, default=['all_true', 'all_true', 'continuous', 'continuous'], nargs='*', help='masks for the model')
parser.add_argument('--factors', type=float, default=[0.25, 0.25, 0.25, 0.25], nargs='*', help='factors for each level')
# for the training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='trial', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='logs', help='directory to save logs')
parser.add_argument('--checkpoint', type=int, default=1, help='save model after each checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='print loss after each epoch')
# linear evaluation
parser.add_argument('--eval', type=str, default='', help='model weight file path to perform linear evaluation. (no pretraining)')
# todo
# parser.add_argument('--resume', type=str, default='', help='resume training from a checkpoint')

args = parser.parse_args()

logdir = os.path.join(args.logdir, f'mcp_{args.data}_{args.seed}')
if not os.path.exists(logdir):
    os.makedirs(logdir)

def main():
    seed_everything(args.seed)
    print(f'=> set seed to {args.seed}')
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.root, args.data, split=args.length)
    
    device = get_device()
    print(f'=> Running on {device}')
    
    model = MCP(
        input_dims=X_test.shape[-1],
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        length=args.length,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        queue_size=args.queue_size,
        num_queue=args.num_queues,
        multi_gpu=args.multi_gpu,
        callback_func=pretrain_callback
    )
    
    if args.eval: # linear evaluation
        if os.path.isfile(args.eval):
            print(f'=> perform linear evaluation on {args.eval}')
            model.load(args.eval)
            
            val_metrics_dict = eval_classification(model, X_train, y_train[:, 0], X_val, y_val[:, 0])
            print('=> Linear evaluation for validation set\n', val_metrics_dict)
            
            test_metrics_dict = eval_classification(model, X_train, y_train[:, 0], X_test, y_test[:, 0])
            print('=> Linear evaluation for test set\n', test_metrics_dict)
        else:
            print(f'=> find nothing in {args.eval}')
    else: # train the model
        loss_list = model.fit(
            X_train,
            y_train,
            shuffle_function=args.shuffle,
            masks=args.masks,
            factors=args.factors,
            epochs=args.epochs,
            verbose=args.verbose
            )
        # save training loss
        np.save(os.path.join(logdir, 'loss.npy'), loss_list)
    
    
def pretrain_callback(model, epoch, checkpoint=args.checkpoint):
    if (epoch+1) % checkpoint == 0:
        model.save(os.path.join(logdir, f'pretrain_{epoch+1}.pth'))
        

def eval_classification(model, train_data, train_labels, test_data, test_labels, fraction=None):
    '''
    Args:
      fraction (Union[float, NoneType]): The fraction of training data. It used to do semi-supervised learning.
    '''

    assert train_labels.ndim == 1 or train_labels.ndim == 2

    if fraction:
        # use first fraction number of training data
        print(f'=> use {fraction} of training data for evaluation')
        train_data = train_data[:int(train_data.shape[0]*fraction)]
        train_labels = train_labels[:int(train_labels.shape[0]*fraction)]
        # print(f"Fraction of train data used for semi_supervised learning:{fraction}\n")

    train_repr = model.encode(train_data)
    test_repr = model.encode(test_data)

    clf = fit_lr(train_repr, train_labels)

    pred_prob = clf.predict_proba(test_repr)
    # print(pred_prob.shape)
    target_prob = (F.one_hot(torch.tensor(test_labels).long(), num_classes=int(train_labels.max()+1))).numpy()
    # print(target_prob.shape)
    pred = pred_prob.argmax(axis=1)
    target = test_labels

    metrics_dict = {}
    metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro', multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')

    return metrics_dict
        

if __name__ == '__main__':
    main()


    
    
    
    
    
    



