#!/usr/bin/env python3

#current layer in Dynamic Unet
def conv2d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, init:LayerFunc=nn.init.kaiming_normal_) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None: padding = ks//2
    return init_default(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)

def conv2d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose2d:
    "Create `nn.ConvTranspose2d` layer."
    return nn.ConvTranspose2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

def relu(inplace:bool=False, leaky:float=None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)

def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, is_1d:bool=False,
               norm_type:Optional[NormType]=NormType.Batch,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)



#############################################################################################################################
#Conv3D from pytorch
#sugested conv3d layer to replace conv2d layer in Dynamic Unet
def conv3d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, init:LayerFunc=nn.init.kaiming_normal_) -> nn.Conv3d:
	"Create and initialise 'nn.Conv3d' layer. "
	if padding is None: padding = ks//2 #check what this line does
	return init_default(nn.Conv3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)

def conv3d_trans(ni:int, nf:int, ks:int=2, stride:int=2, padding:int=0, bias=False) -> nn.ConvTranspose3d:
	"Create 'nn.ConvTranspose3d' layer."
	return(nn.ConvTranspose3d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias))

#relu stays the same

def conv_layers(ni:int, nf:int, ks:int=3, stride:int=1, padding=int=None, bias:bool=Nonem is_1d:bool=False,  
				is_3d:bool=False, norm_type:Optional[NormType]=NormType.Batch, use_activ:bool=True, leaky:float=None, 
				transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
	"Create a sequence of convolutional(ni to nf), ReLU (if use_activ) and batchnorm (if bn) layers."
	if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv3d if is_3d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm3d if is_3d else nn.BatchNorm2d)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


