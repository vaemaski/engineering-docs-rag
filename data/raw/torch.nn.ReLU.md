# ReLU#

Source: https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html

# ReLU


**class torch.nn. ReLU ( inplace = False ) [source]**


Applies the rectified linear unit function element-wise.

ReLU ( x ) = ( x ) + = max ⁡ ( 0 , x ) \text{ReLU}(x) = (x)^+ = \max(0, x) ReLU ( x ) = ( x ) + = max ( 0 , x )

**Parameters :**


inplace ( bool ) – can optionally do the operation in-place. Default: False

**Shape:**

- Input: ( ∗ ) (*) ( ∗ ) , where ∗ * ∗ means any number of dimensions.
- Output: ( ∗ ) (*) ( ∗ ) , same shape as the input.

Examples:

```python
>>> m = nn.ReLU()
  >>> input = torch.randn(2)
  >>> output = m(input)


An implementation of CReLU - https://arxiv.org/abs/1603.05201

  >>> m = nn.ReLU()
  >>> input = torch.randn(2).unsqueeze(0)
  >>> output = torch.cat((m(input), m(-input)))
```


**extra_repr ( ) [source]**


Return the extra representation of the module.

**Return type :**


str

**forward ( input ) [source]**


Runs the forward pass.

**Return type :**


Tensor