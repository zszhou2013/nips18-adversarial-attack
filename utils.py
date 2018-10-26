import torch
from model import WrapperModel

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 5)
        
def load_model(model_name, model_path):
    model = WrapperModel(model_name)
    model.load_state_dict(torch.load(model_path))
    return model

def load_models(device):
    root = 'models/'
    resnet18_adv = load_model('resnet18',root+'resnet18_lr_0.01_adv_16_10_10.pth')
    resnet18 = load_model('resnet18',root+'resnet18_lr_0.01.pth')

    resnet50_adv = load_model('resnet50',root+'resnet50_lr_0.01_adv_16_20_10.pth')
    resnet50 = load_model('resnet50', root+'resnet50_lr_0.01.pth')
    

    resnet18 = resnet18.to(device)
    resnet18_adv = resnet18_adv.to(device)
    resnet50 = resnet50.to(device)
    resnet50_adv = resnet50_adv.to(device) 
    
    return [resnet18, resnet18_adv, resnet50, resnet50_adv]