# https://github.com/buaabai/Ternary-Weights-Network

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from itertools import chain

import model as M
from inq.sgd import ELQSGD, SQ_ELQSGD, INQSGD
from inq.quantization_scheduler import INQScheduler, ELQScheduler,SQ_ELQScheduler_custom_layer, SQ_ELQScheduler_custom_filter
import inq.quantization_scheduler as inqs
import input_pipeline
import time
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_model(model, acc, name):
    print('==>>>Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict()
    }
    torch.save(state, f"trained_models2/{name}")
    print('*** DONE! ***')

def ParseArgs():
    parser = argparse.ArgumentParser(description='Quantisation approaches using multiple algorithms')
    parser.add_argument('--epochs', type=int, default=6, metavar='N', 
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
    parser.add_argument("--test_name", type=str, default="ignore",
                        help="name of test")
    
    args = parser.parse_args()
    return args

def main():
    args = ParseArgs()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = input_pipeline.main(args.dataset)
    
    if args.model == "vgg9":
        model = M.VGG9(10)
        model.load_state_dict(torch.load("vgg9_fwn.pkl")["state_dict"])
    elif args.model == "resnet20":
        model = M.ResNet20(10)
        model.load_state_dict(torch.load("resnet20_fwn.pkl")["state_dict"])
 
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
    elif args.quant == "sq_elq_custom_layer":
        iterative_steps = [0.5, 0.75, 0.875, 1]
        optimizer = SQ_ELQSGD([
            {'params': quantized_parameters},
            {'params': fp_parameters, 'weight_bits': None}
        ], 0.01, momentum=0.9, weight_decay=0.0001, weight_bits=3)
        quantization_scheduler = SQ_ELQScheduler_custom_layer(optimizer, iterative_steps, args.prob_type, args.e_type, strategy="pruning")
    elif args.quant == "sq_elq_custom_filter":
        iterative_steps = [0.5, 0.75, 0.875, 1]
        optimizer = SQ_ELQSGD([
            {'params': quantized_parameters},
            {'params': fp_parameters, 'weight_bits': None}
        ], 0.01, momentum=0.9, weight_decay=0.0001, weight_bits=3)
        quantization_scheduler = SQ_ELQScheduler_custom_filter(optimizer, iterative_steps, args.prob_type, args.e_type, strategy="pruning")
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

    start = time.time()

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

    elapsed = time.time() - start

    save_model(model, acc, f"model_{args.test_name}_{args.quant}_{args.model}_{args.dataset}_{args.e_type}_{args.prob_type}.pkl")

    d = {"model": args.model, "quant": args.quant, "prob_type": args.prob_type, 
         "e_type": args.e_type, "dataset": args.dataset,
         "final_acc": float(f"{float(acc):.2f}"), "total_epoch": args.epochs,
         "time": float(f"{elapsed/60:.2f}")}

    with open(f"txt_results/final2.txt", 'a+') as f:
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


if __name__ == '__main__':
    main()