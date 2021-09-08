import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
from torchvision.models import vgg

from .models import register_model

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class Bilinear(nn.Module):
  #kanei to upsampling 
  #ftiaxnei ena filtro
  #upsampling mesw kernel, convtranspose2d
  #to kernel (weight) den ginetai optimise
  #to upologizw me linear interpolations apo ta geitonika

    def __init__(self, factor, num_channels):
        super().__init__()
        self.factor = factor #poso thelw na anevasw ti diastasi tis eikonas 
        filter = get_upsample_filter(factor * 2)
        w = torch.zeros(num_channels, num_channels, factor * 2, factor * 2)
        for i in range(num_channels):
            w[i, i] = filter
        self.register_buffer('w', w)

    def forward(self, x):
        return F.conv_transpose2d(x, Variable(self.w), stride=self.factor)


@register_model('fcn8s')
class VGG16_FCN8s(nn.Module):

#VGG: (pre-trained)
#during training the input to the conv is a fixed size 224 x 224 RGB image
#the inpy processing we do is subtracting the mean RGB value computed on the training set
#vgg: the image is passed through a stack of conv layers where we use filters with a very small
#receptive field 3x3 (which is the smallest size to capture the notion of left/right, up/down, center)
#1x1 convolution filters (if applicable?) which can be seen as a linear transformation of the input channels
#(followed by non-linearity).
#the conv stride is fixed to 1 pixel, the spatial padding of conv layer input is such that the spatial resolution is
#preserved after convolution, i.e. the padding is 1 pixel for 3x3 conv layers.
#spatial pooling is carried out by 5 max-pooling layers, which follow some of the conv layers (not all the conv layers are
#followed by max pooling)


#1x1 convolution: a way to increase the non linearity without affecting the receptive fields of the conv layers.
#a linear projection onto the space of the same dimensionality
#(the number of input and output channels is the same), an additional non-linearity is introduced



#transform 1: anti gi auto to transformation, orizw kai deutero
#to deutero transformation tha exei gia mesi timi kai diaspora ta 0.5


#this was transform before, i made it 'transform1'
    transform1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ]) #eikones einai sto [0,1] kai afairw ti mesi timi (konta sto 0.5)
        #kai diairw me std (konta sto 0.2): 1st transformation
        #tha dwsei tis eikones se ena range ligo megalitero apo to [-1,1] px [-2,2]

        
    #i added this, independent transformation from the transform1
    #transform2 maps the images to [-1,1] as MUNIT is trained in this range 
    transform2 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
        ])


        #prwta ToTensor, meta normalise me mean kai variances apo pragmatikes eikones
        #anti mean=0.5 kai st dev=0.5 
#transform 2: me mesi timi 0.5 kai st dev 0.5


    def __init__(self, num_cls=19, pretrained=True, weights_init=None, 
            output_last_ft=False):
        super().__init__()
        self.output_last_ft = output_last_ft
        self.vgg = make_layers(vgg.cfgs['D']) #cnn apo to vgg  #cfg itan prin
        self.vgg_head = nn.Sequential( #prosthetoun epipleon convolutional sto vgg network gia to fcnn
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, num_cls, 1) #extra regularisation, teleutaio layer, 4094 channels , feature map mporei na einai polu mikro 
            ) #4096 channels telika, tha ta kanei map ston arithmo twn klasewn
        #stacking more conv layers with activations functions on them
        #makes the decision function more discriminative

        #upsampling sto output tou vgg
        #kai sto output tou proteleutaiou layer kanei upsampling   
        self.upscore2 = self.upscore_pool4 = Bilinear(2, num_cls) #upsampling sto output tou vgg_head kai sto output tou proteleutaiou layer tou vgg_head
        self.upscore8 = Bilinear(8, num_cls)
        self.score_pool4 = nn.Conv2d(512, num_cls, 1) #input 512 channels
        #pairnei to output tou self.vgg 
        for param in self.score_pool4.parameters():
            init.constant_(param, 0) #apo polla levels kanei upsampling
        self.score_pool3 = nn.Conv2d(256, num_cls, 1)
        for param in self.score_pool3.parameters():
            init.constant_(param, 0) #concatenate, gia na mi xasei arketi xwriki pliroforia
        
        if pretrained:
            if weights_init is not None: #random initialisation i na fortwsei ta weights apo to vgg
                self.load_weights(torch.load(weights_init))
            else:
                self.load_base_weights()
 
    def load_base_vgg(self, weights_state_dict):
        vgg_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg.')
        self.vgg.load_state_dict(vgg_state_dict)
     
    def load_vgg_head(self, weights_state_dict):
        vgg_head_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg_head.') 
        self.vgg_head.load_state_dict(vgg_head_state_dict)
    
    def get_dict_by_prefix(self, weights_state_dict, prefix):
        return {k[len(prefix):]: v 
                for k,v in weights_state_dict.items()
                if k.startswith(prefix)}


    def load_weights(self, weights_state_dict):
        self.load_base_vgg(weights_state_dict)
        self.load_vgg_head(weights_state_dict)

    def split_vgg_head(self):
        self.classifier = list(self.vgg_head.children())[-1]
        self.vgg_head_feat = nn.Sequential(*list(self.vgg_head.children())[:-1])


    def forward(self, x):
        input = x
        x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0) #padding 
        intermediates = {}
        fts_to_save = {16: 'pool3', 23: 'pool4'}
        for i, module in enumerate(self.vgg):  #ksekinaei me to vgg 
            x = module(x)
            if i in fts_to_save:
                intermediates[fts_to_save[i]] = x
       
        ft_to_save = 5 # Dropout before classifier
        last_ft = {}
        for i, module in enumerate(self.vgg_head): #meta to vgg_head 
            x = module(x)
            if i == ft_to_save:
                last_ft = x      
        
        _, _, h, w = x.size()
        upscore2 = self.upscore2(x)
        pool4 = intermediates['pool4']
        score_pool4 = self.score_pool4(0.01 * pool4)
        score_pool4c = _crop(score_pool4, upscore2, offset=5)
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        pool3 = intermediates['pool3']
        score_pool3 = self.score_pool3(0.0001 * pool3)
        score_pool3c = _crop(score_pool3, upscore_pool4, offset=9)
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)
        score = _crop(upscore8, input, offset=31)
        if self.output_last_ft: 
            return score, last_ft
        else:
            return score


    def load_base_weights(self):
        """This is complicated because we converted the base model to be fully
        convolutional, so some surgery needs to happen here."""
        base_state_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
        vgg_state_dict = {k[len('features.'):]: v
                          for k, v in base_state_dict.items()
                          if k.startswith('features.')}
        self.vgg.load_state_dict(vgg_state_dict)
        vgg_head_params = self.vgg_head.parameters()
        for k, v in base_state_dict.items():
            if not k.startswith('classifier.'):
                continue
            if k.startswith('classifier.6.'):
                # skip final classifier output
                continue
            vgg_head_param = next(vgg_head_params)
            vgg_head_param.data = v.view(vgg_head_param.size())


def init_eye(tensor):
    if isinstance(tensor, Variable):
        init_eye(tensor.data)
        return tensor
    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


def _crop(input, shape, offset=0):
    _, _, h, w = shape.size()
    return input[:, :, offset:offset + h, offset:offset + w].contiguous()


def make_layers(cfg, batch_norm=False):
    """This is almost verbatim from torchvision.models.vgg, except that the
    MaxPool2d modules are configured with ceil_mode=True.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            modules = [conv2d, nn.ReLU(inplace=True)]
            if batch_norm:
                modules.insert(1, nn.BatchNorm2d(v))
            layers.extend(modules)
            in_channels = v
    return nn.Sequential(*layers)
