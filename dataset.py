import torch
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image 
import numpy as np    


def load_wnids(path='../tiny-imagenet-200/wnids.txt'):
    with open(path) as f:
        wnids = f.readlines()
        assert len(wnids) == 200
        wnids = [x.strip() for x in wnids]
    class2idx = {wnids[i]:i  for i in range(len(wnids))}
    return wnids, class2idx

class TinyTrainset(Dataset):
    
    def __init__(self, root, wnids_path, transform=None):
        super(TinyTrainset, self).__init__()
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        image_lists = [glob.glob(os.path.join(root, cls, 'images','*.JPEG'))  for cls in classes]
        
        self.images = []
        for image_list in image_lists:
            self.images += image_list
        labels = [ image.split('/')[-1].split('_')[0] for image in self.images]   
        
        self.wnids, self.class_to_idx = load_wnids(wnids_path)
        self.idxs = [ self.class_to_idx[label] for label in labels]
        self.labels = labels
        self.transform = transform
        
        assert(len(self.wnids)==200 and len(self.images)==200*500)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):        
        image = Image.open(self.images[index]).convert('RGB')
        label_idx = self.idxs[index]
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx
        

class TinyValset(Dataset):
    
    def __init__(self, root, wnids_path, ann_path, transform=None):
        super(TinyValset, self).__init__()
        
        self.root = root
        self.transform = transform
        self.wnids, self.class_to_idx = load_wnids(wnids_path)
        
        with open(ann_path) as f:
            labels = f.readlines()
            assert len(labels) == 10000        
            data = [ label.split('\t')[:2] for label in labels]
            self.images, self.image_labels = zip(*data)        
        self.idxs = [ self.class_to_idx[label] for label in self.image_labels]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):        
        image = Image.open(os.path.join(self.root, self.images[index])).convert('RGB')
        label_idx = self.idxs[index]
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx        
        
def test_trainset():
    print('test train set ')
    root = '../tiny-imagenet-200/train/'
    wnids_path = '../tiny-imagenet-200/wnids.txt'
    
    trainset = TinyTrainset(root, wnids_path)
    idx = np.random.randint(len(trainset))
    sample = trainset[idx]
    image, label = sample
    image = np.array(image)
    assert image.shape == (64, 64, 3)
    assert 0 <= label and label < 200
    
    print(trainset.images[idx], trainset.labels[idx], trainset.idxs[idx])
    
def test_valset():
    print('test val set')
    root = '../tiny-imagenet-200/val/images/'
    wnids_path = '../tiny-imagenet-200/wnids.txt'
    ann_path = '../tiny-imagenet-200/val/val_annotations.txt'
    
    valset = TinyValset(root, wnids_path, ann_path)
    idx = np.random.randint(len(valset))
    
    image, label = valset[idx]
    image = np.array(image)
    
    assert image.shape == (64, 64, 3)
    assert 0 <= label and label < 200    
    
    print(valset.images[idx], valset.image_labels[idx], valset.idxs[idx])
    
if __name__ == '__main__':   
    print('test')
    test_trainset()
    test_valset()
    print('ok!')