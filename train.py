import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import WrapperModel
from dataset import TinyTrainset, TinyValset

import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from utils import AverageMeter
import argparse


train_root = '../tiny-imagenet-200/train/'
val_root = '../tiny-imagenet-200/val/images/'
val_ann = '../tiny-imagenet-200/val/val_annotations.txt'
wnids_path = '../tiny-imagenet-200/wnids.txt'

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--gpu', default='1', type=str, help='gpu id')
parser.add_argument('--num_workers', default=8, type=int, help='num_workers')
parser.add_argument('--pretrained', '-p', action='store_true', help='model pretrained on imagenet')
parser.add_argument('--model_name', default='resnet50', type=str, help='model name')


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    losses = AverageMeter()
    acc = AverageMeter()    

    for batch in tqdm(dataloader):
        input, label = batch
        input = input.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            output = model(input)

            loss = criterion(output, label)

            correct = torch.sum(output.max(1)[1] == label)  
            acc.update(correct.item() / label.size(0), label.size(0))
            losses.update(loss.item() / label.size(0), label.size(0))
    print(f'eval: epoch{epoch}, loss:{losses.avg}, acc:{acc.avg} ') 
    return acc.avg, losses.avg
    

def train(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    acc = AverageMeter()    
    for batch in tqdm(data_loader):
        input, label = batch
        input = input.to(device)
        label = label.to(device)         

        output = model(input)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.sum(output.max(1)[1] == label)  
        acc.update(correct.item() / label.size(0), label.size(0))
        losses.update(loss.item() / label.size(0), label.size(0))
    print(f'train: epoch{epoch}, loss:{losses.avg}, acc:{acc.avg} ') 
    return acc.avg, losses.avg


def main():
    args = parser.parse_args()
    args.tag = ''
    if args.pretrained:
        args.tag += 'pretrained'
        
    args.tag += f'lr_{args.lr}'
    
    print(args)
    
    log_path = f'logs/{args.model_name}_{args.tag}/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    
    device = torch.device(f'cuda:{args.gpu}')
    
    normlize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normlize  # use or not
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        #normlize  # use or not
    ])

    trainset = TinyTrainset(train_root, wnids_path, transform=transform_train) 
    valset = TinyValset(val_root, wnids_path, val_ann, transform=transform_val)


    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    model = WrapperModel(args.model_name, pretrained=args.pretrained)
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = f'saved_models/base_models/{args.model_name}_{args.tag}.pth'
    for epoch in range(args.epochs):
        train_acc, train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_acc, val_loss = validate(model,val_loader, criterion, device, epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'best_acc:{best_acc}, save to {save_path}')
            torch.save(model.state_dict(), save_path)

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
    
        
if __name__ == '__main__':
    main()