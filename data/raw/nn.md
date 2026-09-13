# torch.nn#

Source: https://docs.pytorch.org/docs/2.14/nn.html

# torch.nn


Created On: Dec 23, 2016 | Last Updated On: Jun 02, 2026

These are the basic building blocks for graphs:

## Containers


Global Hooks For Module

## Convolution Layers


## Pooling layers


## Padding Layers


## Non-linear Activations (weighted sum, nonlinearity)


## Non-linear Activations (other)


## Normalization Layers


## Recurrent Layers


## Transformer Layers


## Linear Layers


## Dropout Layers


## Sparse Layers


## Distance Functions


## Loss Functions


## Vision Layers


## Shuffle Layers


## DataParallel Layers (multi-GPU, distributed)


## Utilities


From the torch.nn.utils module:

Utility functions to clip parameter gradients.

Utility functions to flatten and unflatten Module parameters to and from a single vector.

Utility functions to fuse Modules with BatchNorm modules.

Utility functions to convert Module parameter memory formats.

Utility functions to apply and remove weight normalization from Module parameters.

Utility functions for initializing Module parameters.

Utility classes and functions for pruning Module parameters.

Parametrizations implemented using the new parametrization functionality
in torch.nn.utils.parameterize.register_parametrization() .

Utility functions to parametrize Tensors on existing Modules.
Note that these functions can be used to parametrize a given Parameter
or Buffer given a specific function that maps from an input space to the
parametrized space. They are not parameterizations that would transform
an object into a parameter. See the Parametrizations tutorial for more information on how to implement your own parametrizations.

Utility functions to call a given Module in a stateless manner.

Utility functions in other modules

## Quantized Functions


Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than
floating point precision. PyTorch supports both per tensor and per channel asymmetric linear quantization. To learn more how to use quantized functions in PyTorch, please refer to the Quantization documentation.

## Lazy Modules Initialization