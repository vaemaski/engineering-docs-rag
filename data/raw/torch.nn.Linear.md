# Linear#

Source: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html

# Linear


**class torch.nn. Linear ( in_features , out_features , bias = True , device = None , dtype = None ) [source]**


Applies an affine linear transformation to the incoming data: y = x A T + b y = xA^T + b y = x A T + b .

This module supports TensorFloat32 .

On certain ROCm devices, when using float16 inputs this module will use different precision for backward.

**Parameters :**

- in_features ( int ) – size of each input sample
- out_features ( int ) – size of each output sample
- bias ( bool ) – If set to False , the layer will not learn an additive bias.
Default: True

**Shape:**

- Input: ( ∗ , H in ) (*, H_\text{in}) ( ∗ , H in ​ ) where ∗ * ∗ means any number of
dimensions including none and H in = in_features H_\text{in} = \text{in\_features} H in ​ = in_features .
- Output: ( ∗ , H out ) (*, H_\text{out}) ( ∗ , H out ​ ) where all but the last dimension
are the same shape as the input and H out = out_features H_\text{out} = \text{out\_features} H out ​ = out_features .

**Variables :**

- weight ( torch.Tensor ) – the learnable weights of the module of shape ( out_features , in_features ) (\text{out\_features}, \text{in\_features}) ( out_features , in_features ) . The values are
initialized from U ( − k , k ) \mathcal{U}(-\sqrt{k}, \sqrt{k}) U ( − k ​ , k ​ ) , where k = 1 in_features k = \frac{1}{\text{in\_features}} k = in_features 1 ​
- bias – the learnable bias of the module of shape ( out_features ) (\text{out\_features}) ( out_features ) .
If bias is True , the values are initialized from U ( − k , k ) \mathcal{U}(-\sqrt{k}, \sqrt{k}) U ( − k ​ , k ​ ) where k = 1 in_features k = \frac{1}{\text{in\_features}} k = in_features 1 ​

Examples:

```python
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
```


**extra_repr ( ) [source]**


Return the extra representation of the module.

**Return type :**


str

**forward ( input ) [source]**


Runs the forward pass.

**Return type :**


Tensor

**reset_parameters ( ) [source]**


Resets parameters based on their initialization used in __init__ .