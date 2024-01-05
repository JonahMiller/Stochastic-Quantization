# https://github.com/buaabai/Ternary-Weights-Network

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets,  transforms
import argparse
from itertools import chain

import model as M
import util as U
from inq.sgd import ELQSGD, SQ_ELQSGD, INQSGD
from inq.quantization_scheduler import INQScheduler, ELQScheduler, SQ_ELQScheduler
import inq.quantization_scheduler as inqs
import input_pipeline
import time
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def ParseArgs():
    parser = argparse.ArgumentParser(description='Quantisation approaches using multiple algorithms')
    parser.add_argument('--epochs', type=int, default=4, metavar='N', 
                        help='number of epoch to train(default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', 
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model',  type=str,  default="vgg9", 
                        help='which model do you want to use')
    parser.add_argument("--quant", type=str, default="elq",
                        help="which quantization method to use")
    parser.add_argument("--num_workers", type=int, default="6",
                        help="how many workers to use")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="which dataset to use")
    parser.add_argument("--prob_type", type=str, default="linear",
                        help="which probability function to determine quantization")
    parser.add_argument("--e_type", type=str, default="default",
                        help="how to deal with error function")
    
    args = parser.parse_args()
    return args

def main():
    args = ParseArgs()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = input_pipeline.main(args.dataset)
        
    model = M.VGG9(10)
    model.load_state_dict(torch.load("model_state.pkl")["state_dict"])
 
    quantized_parameters = list(chain.from_iterable([[p for n, p in m.named_parameters() if 'weight' in n] for m in model.modules() 
                                                      if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear)]))
 
    fp_parameters1 = list(chain.from_iterable([[p for n, p in m.named_parameters() if 'bias' in n] for m in model.modules() 
                                                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear)]))
 
    fp_parameters2 = list(chain.from_iterable([[p for n, p in m.named_parameters()] for m in model.modules() 
                                                if isinstance(m,nn.BatchNorm2d)]))
    
    fp_parameters = fp_parameters1 + fp_parameters2


    if args.quant == "elq":
        iterative_steps = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0]
        optimizer = ELQSGD([
            {'params': quantized_parameters},
            {'params': fp_parameters, 'weight_bits': None}
        ], 0.01, momentum=0.9, weight_decay=0.0001, weight_bits=3)
        quantization_scheduler = ELQScheduler(optimizer, iterative_steps, strategy="pruning")
    elif args.quant == "sq_elq":
        iterative_steps = [0.5, 0.75, 0.875, 1]
        optimizer = SQ_ELQSGD([
            {'params': quantized_parameters},
            {'params': fp_parameters, 'weight_bits': None}
        ], 0.01, momentum=0.9, weight_decay=0.0001, weight_bits=3)
        quantization_scheduler = SQ_ELQScheduler(optimizer, iterative_steps, args.prob_type, args.e_type, strategy="pruning")
    elif args.quant == "inq":
        iterative_steps = [0.5, 0.75, 0.875, 1]
        optimizer = INQSGD([
            {'params': quantized_parameters},
            {'params': fp_parameters, 'weight_bits': None}
        ], 0.001, momentum=0.9, weight_decay=0.0003, weight_bits=3)
        quantization_scheduler = INQScheduler(optimizer, iterative_steps, strategy="pruning")

    quantization_epochs = len(iterative_steps)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2,
                                                gamma=0.5)

    for _ in range(quantization_epochs):
        inqs.reset_lr_scheduler(scheduler)
        quantization_scheduler.step()
    
        for epoch_index in range(1, args.epochs + 1):

            train(args, epoch_index, train_loader, model, optimizer, criterion)
            acc, loss = test(args, model, test_loader, criterion)

            scheduler.step()

    if args.quant == "elq":
        quantization_scheduler.finish_quantize()
        acc, loss = test(args, model, test_loader, criterion)

    d = {"model": f"{args.model}", "quant": f"{args.quant}", "dataset": f"{args.dataset}", 
         "prob_type": f"{args.prob_type}", "e_type": f"{args.e_type}", "final_acc": f"{float(acc):.2f}"}

    with open(f"txt_results/inq_test_run_1.txt", 'a+') as f:
        f.write(str(d) + "\n")
        f.write("\n")


def train(args, epoch_index, train_loader, model, optimizer, criterion):
    model.train()
    train_time = time.time()
    for batch_idx, (input, label) in enumerate(train_loader):

        input, label = Variable(input).to(device), Variable(label).to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input)
        loss = criterion(output, label)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index,  batch_idx * len(input),  len(train_loader.dataset), 
                100. * batch_idx / len(train_loader),  loss.data))
    print(f"Train Epoch: {epoch_index} took {(time.time() - train_time):.2f} seconds")

def test(args, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for input, label in test_loader:

        input, label = Variable(input).to(device), Variable(label).to(device)
        output = model(input)

        test_loss += criterion(output, label).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f},  Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,  correct,  len(test_loader.dataset), 
        100. * correct / len(test_loader.dataset)))
    
    return acc, test_loss

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return f'model size: {size_all_mb:.3f}MB'

if __name__ == '__main__':
    main()