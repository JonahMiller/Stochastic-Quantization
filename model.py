import sys

sys.path.append("..")
from models.vgg import VGG9 as v
from models.resnet import resnet20 as r

def VGG9(num_classes):
    return v(input_size=32, num_class=num_classes)

def ResNet20(num_classes):
    return r(num_class=num_classes)

def count_parameters(model):
    return (sum(p.numel() for p in model.parameters() if p.requires_grad))

if __name__ == "__main__":
    print(count_parameters(VGG9(10)))
    print(count_parameters(ResNet20(10)))