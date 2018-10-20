import torch
import torch.nn as nn
import pretrainedmodels

# current supported models
model_names = ['densenet','resnet18', 'resnet50', 'se_resnet50', 'se_resnext50_32x4d']


class WrapperModel(nn.Module):
    
    def __init__(self, base_model_name, pretrained=True):
        super(WrapperModel, self).__init__()
        
        assert(base_model_name in model_names)
        
        if pretrained:
            base_model = pretrainedmodels.__dict__[base_model_name](pretrained='imagenet')
        else:
            base_model = pretrainedmodels.__dict__[base_model_name](pretrained=None)
            
        
        self.features = nn.Sequential(*list(base_model.children())[:-2])        
        
        feature_num = base_model.last_linear.in_features
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.last_linear = nn.Linear(feature_num, 200)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.last_linear(x)
        return x

def test_model():    
    image = torch.randn(1,3,64,64)
    for model_name in model_names:
        model = WrapperModel(model_name)
        out = model(image)
        print(model_name,image.shape, out.shape)
        
if __name__ == '__main__':
    print('test')
    test_model()
    print('ok')