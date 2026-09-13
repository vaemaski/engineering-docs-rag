# Conv2d#

Source: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html

# Conv2d


**class torch.nn. Conv2d ( in_channels , out_channels , kernel_size , stride = 1 , padding = 0 , dilation = 1 , groups = 1 , bias = True , padding_mode = 'zeros' , device = None , dtype = None ) [source]**


Applies a 2D convolution over an input signal composed of several input
planes.

In the simplest case, the output value of the layer with input size ( N , C in , H , W ) (N, C_{\text{in}}, H, W) ( N , C in ​ , H , W ) and output ( N , C out , H out , W out ) (N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}}) ( N , C out ​ , H out ​ , W out ​ ) can be precisely described as:

where ⋆ \star ⋆ is the valid 2D cross-correlation operator, N N N is a batch size, C in C_{\text{in}} C in ​ and C out C_{\text{out}} C out ​ correspond to in_channels and out_channels respectively, H H H and W W W are the input height and width in pixels.
See the Shape section below for how H out H_{\text{out}} H out ​ and W out W_{\text{out}} W out ​ are derived from kernel_size , stride , padding , and dilation .

This module supports TensorFloat32 .

On certain ROCm devices, when using float16 inputs this module will use different precision for backward.
- stride controls the stride for the cross-correlation, a single
number or a tuple.
- padding controls the amount of padding applied to the input. It
can be either a string {‘valid’, ‘same’} or an int / a tuple of ints giving the
amount of implicit padding applied on both sides.
- dilation controls the spacing between the kernel points; also
known as the à trous algorithm. It is harder to describe, but this link has a nice visualization of what dilation does.
- groups controls the connections between inputs and outputs. in_channels and out_channels must both be divisible by groups . For example, At groups=1, all inputs are convolved to all outputs. At groups=2, the operation becomes equivalent to having two conv
layers side by side, each seeing half the input channels
and producing half the output channels, and both subsequently
concatenated. At groups= in_channels , each input channel is convolved with
its own set of filters (of size out_channels in_channels \frac{\text{out\_channels}}{\text{in\_channels}} in_channels out_channels ​ ).

The parameters kernel_size , stride , padding , dilation can either be:

Note

When groups == in_channels and out_channels == K * in_channels ,
where K is a positive integer, this operation is also known as a “depthwise convolution”.

In other words, for an input of size ( N , C i n , L i n ) (N, C_{in}, L_{in}) ( N , C in ​ , L in ​ ) ,
a depthwise convolution with a depthwise multiplier K can be performed with the arguments ( C in = C in , C out = C in × K , . . . , groups = C in ) (C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in}) ( C in ​ = C in ​ , C out ​ = C in ​ × K , ... , groups = C in ​ ) .

Note

In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting torch.backends.cudnn.deterministic = True . See Reproducibility for more information.

Note

padding='valid' is the same as no padding. padding='same' pads
the input so the output has the shape as the input. However, this mode
doesn’t support any stride values other than 1.

Note

This module supports complex data types i.e. complex32, complex64, complex128 .

**Parameters :**

- in_channels ( int ) – Number of channels in the input image
- out_channels ( int ) – Number of channels produced by the convolution
- kernel_size ( int or tuple ) – Size of the convolving kernel
- stride ( int or tuple , optional ) – Stride of the convolution. Default: 1
- padding ( int , tuple or str , optional ) – Padding added to all four sides of
the input. Default: 0
- dilation ( int or tuple , optional ) – Spacing between kernel elements. Default: 1
- groups ( int , optional ) – Number of blocked connections from input
channels to output channels. Default: 1
- bias ( bool , optional ) – If True , adds a learnable bias to the
output. Default: True
- padding_mode ( str , optional ) – 'zeros' , 'reflect' , 'replicate' or 'circular' . Default: 'zeros'

**Shape:**

- Input: ( N , C i n , H i n , W i n ) (N, C_{in}, H_{in}, W_{in}) ( N , C in ​ , H in ​ , W in ​ ) or ( C i n , H i n , W i n ) (C_{in}, H_{in}, W_{in}) ( C in ​ , H in ​ , W in ​ )
- Output: ( N , C o u t , H o u t , W o u t ) (N, C_{out}, H_{out}, W_{out}) ( N , C o u t ​ , H o u t ​ , W o u t ​ ) or ( C o u t , H o u t , W o u t ) (C_{out}, H_{out}, W_{out}) ( C o u t ​ , H o u t ​ , W o u t ​ ) , where H o u t = ⌊ H i n + 2 × padding [ 0 ] − dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) − 1 stride [ 0 ] + 1 ⌋ H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
          \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor H o u t ​ = ⌊ stride [ 0 ] H in ​ + 2 × padding [ 0 ] − dilation [ 0 ] × ( kernel_size [ 0 ] − 1 ) − 1 ​ + 1 ⌋ W o u t = ⌊ W i n + 2 × padding [ 1 ] − dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) − 1 stride [ 1 ] + 1 ⌋ W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
          \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor W o u t ​ = ⌊ stride [ 1 ] W in ​ + 2 × padding [ 1 ] − dilation [ 1 ] × ( kernel_size [ 1 ] − 1 ) − 1 ​ + 1 ⌋

**Variables :**

- weight ( Tensor ) – the learnable weights of the module of shape ( out_channels , in_channels groups , (\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, ( out_channels , groups in_channels ​ , kernel_size[0] , kernel_size[1] ) \text{kernel\_size[0]}, \text{kernel\_size[1]}) kernel_size[0] , kernel_size[1] ) .
The values of these weights are sampled from U ( − k , k ) \mathcal{U}(-\sqrt{k}, \sqrt{k}) U ( − k ​ , k ​ ) where k = g r o u p s C in ∗ ∏ i = 0 1 kernel_size [ i ] k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]} k = C in ​ ∗ ∏ i = 0 1 ​ kernel_size [ i ] g ro u p s ​
- bias ( Tensor ) – the learnable bias of the module of shape
(out_channels). If bias is True ,
then the values of these weights are
sampled from U ( − k , k ) \mathcal{U}(-\sqrt{k}, \sqrt{k}) U ( − k ​ , k ​ ) where k = g r o u p s C in ∗ ∏ i = 0 1 kernel_size [ i ] k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]} k = C in ​ ∗ ∏ i = 0 1 ​ kernel_size [ i ] g ro u p s ​

Examples

```python
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)
```