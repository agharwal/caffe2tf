# caffe2tf

This Python library (compatible with Tensorflow version 0.12) enables the use of [Caffe](http://github.com/BVLC/caffe)
prototxt file for creating the equivalent neural network in
[TensorFlow](http://tensorflow.org).
While experimenting with different network architectures and hyperparameters, it is 
convenient to keep track of portable, language independent files encoding this
information, rather than making copies of Python files.
caffe2tf is different from, but complimentary to
[tf-slim](https://github.com/tensorflow/models/tree/master/inception/inception/slim)
and the wonderful library
[caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow); these libraries
allow creation of a network via abstractions within Python code (additionally,
caffe-tensorflow converts Caffe model parameter files to NumPy pickles).

caffe2tf is a work in progress, and currently supports the following layers:
  - Input
  - 2D Convolution
  - Pooling (avg/max)
  - ReLU
  - Dropout
  - Inner Product
  - Softmax
  - Sigmoid
  - Reshape
  - Concat
  - Slice

I am currently adding support for layers needed for my experiments, but
if there is interest from the community, I will be happy to add more!

Support planned for:
  - 3D Convolution
  - Batch Normalization
  - Deconvolution
  - Concatenation

For loading pre-trained parameters, the code currently relies on the NumPy pickle
created by [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) for
the corresponding network parameter file. Note that parsing the original Caffe
file via Protobuf is painfully slow, which is why it is best to rely on the
converted file. _~~Note that this feature is untested. I will update this file once
correctness is verified.~~_

Additionally, the code includes a [modified copy](SUPPORTED.proto) of
[caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)
(hope this isn't a copyright infringement; someone, please let me know!), which
has the unimplemented features commented out. This will be updated as support is
added for additional layers.

# Dependencies

Developed in Python 2.7. Requires the following external Python modules:
  - Google Protobuf (can be installed via pip)
  - Tensorflow (refer to the [installation instructions page](https://www.tensorflow.org/versions/master/get_started/os_setup.html#download-and-setup))
  - [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) (only for
    using pre-trained weights from Caffe model file; not needed for successful
    compilation)

# Installation

Download [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto),
and compile using `protoc`, which should be included in Google Protobuf.

```sh
protoc --python_out=. caffe.proto
```

This will create `caffe_pb2.py` in the current directory; move it inside the
`caffe2tf` directory, along with the rest of the source code. That's it!

# Usage

```python
from caffe2tf.network import Network
import tensorflow as tf
# Create network.
net = Network("the_proto_file.prototxt")
# Load variables (requires active Tensorflow session)
with tf.Session() as sess:
    net.load("the_model_pickle_file", sess)
```

To access the output of a particular layer, use the `top` value as defined in the
prototxt file for the model. For example:

```python
# Example for VGG-16.
output = net.layer_output("conv_1_1")
```

# Finally
I am excited to hear your comments, suggestions, and criticisms -- be it regarding
feature requests, bug reports, coding style, or the library architecture!
Feel free to start an `Issues` thread or get in touch directly via email
(ID on my GitHub profile).
