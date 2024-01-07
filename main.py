# https://github.com/buaabai/Ternary-Weights-Network

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets,  transforms, models
import argparse

import model as M
import util as U
import input_pipeline
import time
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def ParseArgs():
    parser = argparse.ArgumentParser(description='Quantisation approaches using multiple algorithms')
    parser.add_argument('--epochs', type=int, default=90, metavar='N', 
                        help='number of epoch to train(default: 90)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', 
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model',  type=str,  default="vgg9", 
                        help='which model do you want to use')
    parser.add_argument("--quant", type=str, default="fwn",
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

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100

    if args.model == "vgg9":
        model = M.VGG9(num_classes)
    elif args.model == "resnet20":
        model = M.ResNet20(num_classes)


    if args.quant == "fwn":
        quantize = U.FWN(model)
    elif args.quant == "bwn":
        quantize = U.BWN(model)
    elif args.quant == "sq_bwn_default_layer":
        quantize = U.SQ_BWN_default_layer(model, args.prob_type)
    elif args.quant == "sq_bwn_custom_layer":
        quantize = U.SQ_BWN_custom_layer(model, args.prob_type, args.e_type)
    elif args.quant == "sq_bwn_custom_filter":
        quantize = U.SQ_BWN_custom_filter(model, args.prob_type, args.e_type)
    elif args.quant == "twn":
        quantize = U.TWN(model)
    elif args.quant == "sq_twn_default_layer":
        quantize = U.SQ_TWN_default_layer(model, args.prob_type)
    elif args.quant == "sq_twn_custom_layer":
        quantize = U.SQ_TWN_custom_layer(model, args.prob_type, args.e_type)
    elif args.quant == "sq_twn_custom_filter":
        quantize = U.SQ_TWN_custom_filter(model, args.prob_type, args.e_type)
    elif args.quant == "tnn":
        quantize = U.Trained_TernarizeOp(model)
    elif args.quant == "sq_tnn":
        quantize = U.SQ_Trained_TernarizeOp(model, args.prob_type, args.e_type)


    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0001)
        
    initial_scaling_factors = [(1.,1.) for m in model.modules() 
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear)]

    # optimizer for updating only scaling factors
    optimizer_sf = optim.Adam([
        Variable(torch.FloatTensor([w_p, w_n]).cuda(), requires_grad=True) 
        for w_p, w_n in initial_scaling_factors
    ], lr=0.00001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.1)

    start = time.time()

    r = [0.5, 0.75, 0.875, 1]
        
    best_acc = 0.0
    best_epoch = 0

    all_acc = []
    all_loss = []

    for epoch_index in range(1, args.epochs + 1):
        if  epoch_index <= int((args.epochs)/4):
            r_it = r[0]
        elif epoch_index <= int(2*(args.epochs)/4):
            r_it = r[1]
        elif epoch_index <= int(3*(args.epochs)/4):
            r_it = r[2]
        else:
            r_it = r[3]

        if args.quant == "tnn" or args.quant == "sq_tnn":
            train_tnn(args, epoch_index, train_loader, model, [optimizer, optimizer_sf], criterion, quantize, r_it)
            acc, loss = test_tnn(args, model, test_loader, criterion, quantize, r_it, scaling_factors=optimizer_sf.param_groups[0]['params'])
        elif args.quant == "elq_twn" or args.quant == "sq_elq_twn":
            train_elq(args, epoch_index, train_loader, model, optimizer, criterion, quantize, r_it)
            acc, loss = test_elq(args, model, test_loader, criterion, quantize, r_it)
        else:
            train_standard(args, epoch_index, train_loader, model, optimizer, criterion, quantize, r_it)
            acc, loss = test_standard(args, model, test_loader, criterion, quantize, r_it)

        all_acc.append(f"{float(acc):.2f}")
        all_loss.append(f"{float(loss):.6f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch_index
            if args.quant == "tnn" or args.quant == "sq_tnn":
                quantize.Quantization(optimizer_sf.param_groups[0]['params'], r_it)
            else:
                quantize.Quantization(r_it)
            U.save_model(model, best_acc, f"model_{args.test_name}_{args.quant}_{args.model}_{args.dataset}_{args.e_type}_{args.prob_type}.pkl")
            quantize.Restore()
        
        scheduler.step()

    elapsed = time.time() - start
    d = {"model": args.model, "quant": args.quant, "prob_type": args.prob_type, "e_type": args.e_type, 
         "best_acc": float(f"{float(best_acc):.2f}"), "dataset": args.dataset,
         "best_epoch": best_epoch, "final_acc": float(f"{float(acc):.2f}"), "total_epoch": args.epochs,
         "time": float(f"{elapsed/60:.2f}")}

    with open(f"txt_results/final.txt", 'a+') as f:
        f.write(str(d) + "\n")
        f.write(str(all_acc) + "\n")
        f.write(str(all_loss) + "\n")
        f.write("\n")


def train_standard(args, epoch_index, train_loader, model, optimizer, criterion, quantize, r):
    model.train()
    train_time = time.time()
    for batch_idx, (input, label) in enumerate(train_loader):

        input, label = Variable(input).to(device), Variable(label).to(device)

        optimizer.zero_grad(set_to_none=True)
        
        quantize.Quantization(r)
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        
        quantize.Restore()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index,  batch_idx * len(input),  len(train_loader.dataset), 
                100. * batch_idx / len(train_loader),  loss.data))
    print(f"Train Epoch: {epoch_index} took {(time.time() - train_time):.2f} seconds")


def train_tnn(args, epoch_index, train_loader, model, optimizer_list, criterion, quantize, r):
    model.train()
    train_time = time.time()

    optimizer, optimizer_sf = optimizer_list

    for batch_idx, (input, label) in enumerate(train_loader):

        input, label = Variable(input).to(device), Variable(label).to(device)

        optimizer.zero_grad(set_to_none=True)
        optimizer_sf.zero_grad(set_to_none=True)

        scaling_factors = optimizer_sf.param_groups[0]['params']
        quantize.Quantization(scaling_factors=scaling_factors, r=r)
        output = model(input)
        loss = criterion(output, label)
        loss.backward()

        quantize.UpdateGradientsAndRestore(scaling_factors)

        optimizer.step()
        optimizer_sf.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index,  batch_idx * len(input),  len(train_loader.dataset), 
                100. * batch_idx / len(train_loader),  loss.data))
    print(f"Train Epoch: {epoch_index} took {(time.time() - train_time):.2f} seconds")


def train_elq(args, epoch_index, train_loader, model, optimizer, criterion, quantize, r):
    model.train()
    train_time = time.time()
    for batch_idx, (input, label) in enumerate(train_loader):

        input, label = Variable(input).to(device), Variable(label).to(device)

        optimizer.zero_grad(set_to_none=True)

        quantize.Quantization(r=r)

        lr = optimizer.param_groups[-1]['lr']
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        
        quantize.UpdateGradientsAndRestore(lr=lr)

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index,  batch_idx * len(input),  len(train_loader.dataset), 
                100. * batch_idx / len(train_loader),  loss.data))
    print(f"Train Epoch: {epoch_index} took {(time.time() - train_time):.2f} seconds")

def test_standard(args, model, test_loader, criterion, quantize, r):
    model.eval()
    test_loss = 0
    correct = 0

    quantize.Quantization(r)
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


def test_tnn(args, model, test_loader, criterion, quantize, r, scaling_factors):
    model.eval()
    test_loss = 0
    correct = 0
    quantize.Quantization(scaling_factors=scaling_factors, r=r)
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


def test_elq(args, model, test_loader, criterion, quantize, r):
    model.eval()
    test_loss = 0
    correct = 0

    quantize.Quantization(r)
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