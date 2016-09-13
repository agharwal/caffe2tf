import caffe_pb2 as cpb
from google.protobuf import text_format
from collections import defaultdict
import tensorflow as tf
import numpy as np
from operator import mul
import logging


class Network:
    # Public functions.
    def __init__(self, proto, input_format="NC*"):
        """Create network using structure from Caffe proto file.

        Args:
            proto: The plaintext network prototxt file on disk.
                Type `str`.
            input_format: The order of inputs specified in the prototxt
                file. The TensorFlow program is expected to provide inputs
                in the order `N*C`, where the dimension `N` denotes batch,
                `C` denotes the number of channels, and `*` denotes the
                rest of the dimensions. Valid arguments are `N*C`, `NC*`,
                `*NC`, `CN*`, `C*N`, and `*CN`. The network will permute
                the order of dimensions specified in prototxt file for the
                InputParameter in the order `N*C`. Defaults to `NC*`.
        """
        # Layer names mapped to outputs.
        self._name_outputs = defaultdict(list)
        # Layer top names mapped to outputs.
        self._top_outputs = defaultdict(list)
        # Layer name to list of variables.
        self._name_vars = defaultdict(list)
        # Protobuf LayerParameter messages for each layer, loaded in the
        # order presented in the proto file.
        self._layer_params = []
        # List of variables.
        self._vars = []
        # Layer name to (name, var) mapping.
        self._layer_vars = defaultdict(list)

        self._input_format = input_format

        # The NetParameter protobuf message.
        self._net_param = cpb.NetParameter()
        # Load and parse the prototxt file.
        self._parse_proto(proto)
        # Create network, based on the parse.
        self._make_network()

    def load(self, model, sess):
        """Load network variables from NumPy file on disk.

        Args:
            model: The path to the model file on disk.
            sess: The TensorFlow session.
        """
        self._load(model, sess)

    def layer_output(self, layer_name):
        if layer_name not in self._top_outputs:
            raise ValueError("`%s` not defined in network" % layer_name)
        return self._top(layer_name)

    # Layer functions.
    def _add_Input(self, lp):
        """Add input layer placeholder to network.

        Does not support undefined shapes.

        Args:
            lp: Caffe LayerParameter protobuf message
                with type "Input".
        """
        def reorder(dims):
            # Figure out the permutation for the shape to convert to N*C order.
            idx_C = self._input_format.index("C")
            if idx_C == 2:
                idx_C = -1
            idx_N = self._input_format.index("N")
            if idx_N == 2:
                idx_N = -1
            # Reorder.
            new_dim = [dims[idx_N]]
            temp = list(dims)
            temp = [i for j, i in enumerate(dims) if j not in [idx_C, idx_N]]
            new_dim += (temp + [dims[idx_C]])
            return new_dim

        # In case of multiple outputs but one shape specification, repeat.
        if len(lp.top) != len(lp.input_param.shape):
            shapes = lp.input_param.shape * len(lp.top)
        else:
            shapes = lp.input_param.shape
        for (top, shape) in zip(lp.top, shapes):
            output = tf.placeholder(tf.float32,
                                    list(map(int, reorder(shape.dim))))
            self._add_output_to_lists(output, lp.name, top)

    def _add_Convolution(self, lp):
        """Add convolution layer to network.

        Note that only 2-D convolutions are currently supported.

        Args:
            lp: Caffe LayerParameter protobuf message
                with type "Convolution".
        """
        filter_size, padding, strides, out_channels = \
            self._parse_ConvolutionParameter(lp.convolution_param)
        # Get the number of chanels in input. Ensure that the number
        # is consistent for each of the bottom layers.
        in_channels = 0
        for bottom in lp.bottom:
            if in_channels == 0:
                in_channels = self._top(bottom).get_shape()[-1]
            if in_channels != self._top(bottom).get_shape()[-1]:
                raise ValueError("bottom layers for %s must have same "
                                 "channel count" % lp.name)
        # Create convolution filter weights.
        with tf.variable_scope(lp.name):
            weights = self._make_vars(
                "weights",
                filter_size + [in_channels, out_channels],
                lp.phase == cpb.TRAIN,
                lp.name)
            if lp.convolution_param.bias_term:
                biases = self._make_vars(
                    "biases",
                    [out_channels],
                    lp.phase == cpb.TRAIN,
                    lp.name)
        # For each bottom layer, create convolutional output
        # feature map.
        for (top, bottom) in zip(lp.top, lp.bottom):
            input = self._top(bottom)
            # Perform 2D convolution.
            with tf.name_scope(top + "/"):
                # Pad input, if necessary.
                padded_input = tf.pad(input, padding)
                # Convolve.
                output = tf.nn.conv2d(padded_input, weights, strides, "VALID")
                # Add biases, if needed.
                if lp.convolution_param.bias_term:
                    output = tf.nn.bias_add(output, biases)
            # Add to lists.
            self._add_output_to_lists(output, lp.name, top)

    def _add_ReLU(self, lp):
        if len(lp.top) != 1 or len(lp.bottom) != 1:
            raise ValueError(
                "ReLU layers (%s) must have exactly one "
                "top and bottom layer" % lp.name)
        top = lp.top[0]
        bottom = lp.bottom[0]
        with tf.name_scope(top + "/"):
            output = tf.nn.relu(self._top(bottom), name=lp.name)
        self._add_output_to_lists(output, lp.name, top)

    def _add_Pooling(self, lp):
        if len(lp.top) != 1 or len(lp.bottom) != 1:
            raise ValueError(
                "Pooling layers (%s) must have exactly one "
                "top and bottom layer" % lp.name)
        top = lp.top[0]
        bottom = lp.bottom[0]

        pp = lp.pooling_param
        input = self._top(bottom)
        # Recover the pooling parameters.
        ksize, padding, strides = self._parse_PoolingParameter(
            pp, input)
        with tf.name_scope(lp.name):
            # Pad, if necessary.
            if padding is not None:
                input = tf.pad(input, padding)
            if pp.pool == cpb.PoolingParameter.MAX:
                output = tf.nn.max_pool(input, ksize, strides, "VALID")
            elif pp.pool == cpb.PoolingParameter.AVE:
                output = tf.nn.avg_pool(input, ksize, strides, "VALID")
            else:
                raise NotImplementedError(
                    "Unsupported pooling: %s" % pp.pool)
        self._add_output_to_lists(output, lp.name, top)

    def _add_InnerProduct(self, lp):
        if len(lp.top) > 1:
            raise ValueError(
                "InnerProduct layers (%s) must have exactly one "
                "top layer" % lp.name)

        inputs = []
        with tf.name_scope(lp.name + "/"):
            for bottom in lp.bottom:
                input = self._top(bottom)
                shape = input.get_shape().as_list()
                # Get the flattened shape, excluding the index
                # correspnding to batch.
                ndims_flat = reduce(mul, shape[1:], 1)  # NOQA
                inputs.append(tf.reshape(input, [shape[0], ndims_flat]))
            # Concatenate flattened inputs.
            if len(inputs) > 1:
                input = tf.concat(1, inputs)
            else:
                input = inputs[0]

        # Prepare weights and biases.
        ipp = lp.inner_product_param
        input_ndims = input.get_shape().as_list()[-1]
        with tf.variable_scope(lp.name):
            weights = self._make_vars(
                "weights",
                [input_ndims, ipp.num_output],
                lp.phase == cpb.TRAIN,
                lp.name)
            if ipp.bias_term:
                biases = self._make_vars(
                    "biases",
                    [ipp.num_output],
                    lp.phase == cpb.TRAIN,
                    lp.name)

        # Compute output.
        with tf.name_scope(lp.name + "/"):
            if ipp.bias_term:
                output = tf.nn.xw_plus_b(input, weights, biases)
            else:
                output = input * weights
        self._add_output_to_lists(output, lp.name, lp.top[0])

    def _add_Dropout(self, lp):
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(top + "/"):
                output = tf.nn.dropout(self._top(bottom),
                                       1.0 - lp.dropout_param.dropout_ratio,
                                       name=lp.name)
            self._add_output_to_lists(output, lp.name, top)

    def _add_Softmax(self, lp):
        for top, bottom in zip(lp.top, lp.bottom):
            with tf.name_scope(top + "/"):
                output = tf.nn.softmax(self._top(bottom))
            self._add_output_to_lists(output, lp.name, top)

    # Helper functions.
    def _make_vars(self, name, shape, trainable, layer_name=None):
        """Create variables of given `name` and `shape`.

        Args:
            name: Variable name. Type `str`.
            shape: A 1-D `Tensor` of type `int32` representing
                the shape of the variables.
            trainable: `True` if variables are trainable.
            layer_name: Name of layer for which variables will be used.
                Type `str`. Ignored if set to `None`. Defaults to `None`.
        """
        vars = tf.get_variable(name, shape, trainable=trainable)
        self._vars.append(vars)
        if layer_name is not None:
            self._layer_vars[layer_name].append((name, vars))
        return vars

    def _add_output_to_lists(self, output, name, top):
        self._top_outputs[top].append(output)
        self._name_outputs[name].append(
            (top, len(self._top_outputs[top]) - 1, output))

    def _parse_proto(self, proto):
        """Parse the proto file to create network graph

        Args:
            proto: Path of prototxt plaintext file on disk.
                Type `str`.
        """
        with open(proto, "r") as f:
            text_format.Merge(f.read(), self._net_param)
            for layer_param in self._net_param.layer:
                self._layer_params.append(layer_param)

    def _top(self, top):
        return self._top_outputs[top][-1]

    def _make_network(self):
        """Construct the network using the parsed proto file."""
        for layer_param in self._layer_params:
            # Does layer need to be skipped?
            if not self._include_layer(layer_param):
                logging.info("Skipping: %s", layer_param.name)
                continue
            if layer_param.type == "Input":
                self._add_Input(layer_param)
            elif layer_param.type == "Convolution":
                self._add_Convolution(layer_param)
            elif layer_param.type == "ReLU":
                self._add_ReLU(layer_param)
            elif layer_param.type == "Pooling":
                self._add_Pooling(layer_param)
            elif layer_param.type == "InnerProduct":
                self._add_InnerProduct(layer_param)
            elif layer_param.type == "Dropout":
                self._add_Dropout(layer_param)
            elif layer_param.type == "Softmax":
                self._add_Softmax(layer_param)
            else:
                raise NotImplementedError(
                    "Contact the bugger who wrote this.")

    def _parse_ConvolutionParameter(self, cp, dims=2):
        out_channels = cp.num_output

        # Infer input dimensions.
        if len(cp.kernel_size) == 1:
            filter_size = [cp.kernel_size[0]] * dims
        else:
            filter_size = cp.kernel_size

        # Infer padding width.
        if len(cp.pad) == 1:
            padding = [cp.pad[0]] * dims
        else:
            padding = cp.pad
        # TensorFlow allows specification of different paddings
        # at both ends of an input axis, but Caffe assumes it would
        # be the same. Create a TF compatible output.
        padding = [[0, 0]] + [[p, p] for p in padding] + [[0, 0]]

        # Infer stride.
        if len(cp.stride) == 0:
            strides = [1] * dims
        elif len(cp.stride) == 1:
            strides = [cp.stride[0]] * dims
        else:
            strides = cp.stride
        strides = [1] + strides + [1]

        return filter_size, padding, strides, out_channels

    def _parse_PoolingParameter(self, pp, bottom):
        # Input Tensor dimensions, ignoring batch size and number of channels.
        ndims = len(bottom.get_shape()) - 2
        # If pooling is global.
        if pp.global_pooling:
            # Kernel size equals to full extent of feature map.
            ksize = bottom.get_shape().as_list()[1:-1]
            # No padding.
            padding = None
            # Fixed stride.
            strides = [1] * (ndims + 2)
            return ksize, padding, strides

        # Figure out kernel size.
        # a) kernel_h and kernel_w are set. Feature dimensions
        #    must be equal to 2.
        if pp.kernel_size is None:
            if ndims != 2:
                raise ValueError(
                    "Either kernel_size or both kernel_x and kernel_y"
                    " must be set.")
            ksize = [pp.kernel_h, pp.kernel_w]
        # b) kernel_size is set
        else:
            ksize = [pp.kernel_size] * ndims
        # Adjust for TensorFlow specs. Kernel should be 1 layer
        # wide for batch and channel Tensor indices.
        ksize = [1] + ksize + [1]

        # Figure out stride.
        # a) stride is set
        if pp.stride > 0:
            strides = [pp.stride] * ndims
        # b) stide_h and stride_w are set (we assume both).
        else:
            strides = [pp.stride_h, pp.stride_w]
        # Adjust for TensorFlow specs. Max should be computed
        # over each feature map for each instance in the batch.
        strides = [1] + strides + [1]

        # Infer padding width. No padding along batch and channel
        # indices
        if pp.pad > 0:
            padding = [0] + ([pp.pad] * ndims) + [0]
        elif pp.pad_h > 0 or pp.pad_w > 0:
            padding = [0, pp.pad_h, pp.pad_w, 0]
        else:
            padding = None
        # TensorFlow allows specification of different paddings
        # at both ends of an input axis, but Caffe assumes it would
        # be the same. Create a TF compatible output.
        if padding is not None:
            padding = [[p, p] for p in padding]

        return ksize, padding, strides

    def _load(self, model, sess):
        # Load data from file.
        data_dict = np.load(model).item()
        # For each layer type, load data.
        for layer_proto in self._layer_params:
            layer_name = layer_proto.name
            # Make sure layer has variables.
            if not self._has_vars(layer_proto):
                continue
            # It is possible that the file is missing data
            # for some layers. Warn the user.
            if layer_name not in data_dict:
                logging.warning("No variables found for layer: %s",
                                layer_name)
                continue
            # Load each variable in layer.
            for (name, vars) in self._layer_vars[layer_name]:
                # Still possible for stuff to be missing. This is unusual.
                # Warn the poor soul.
                if name not in data_dict[layer_name]:
                    logging.warning("Variables %s not found for layer %s",
                                    name,
                                    layer_name)
                    continue
                try:
                    # Time to assign values.
                    sess.run(vars.assign(data_dict[layer_name][name]))
                except ValueError as e:
                    # Something is very wrong. User likely has the wrong
                    # model file.
                    logging.warning("Dimension mismatch for variables"
                                    "%s in layer %s. %s.",
                                    name,
                                    layer_name,
                                    str(e))

    def _has_vars(self, layer_proto):
        return layer_proto.name in self._layer_vars

    def _satisfied_NSR(self, nsr):
        """Check if `NetStateRule` is satisfied based on `NetState`.

        Args:
            nsr: `NetStateRule` protobuf message.

        Returns:
            Boolean `True` iff `NetState` agrees with `nsr`.
        """
        np_state = self._net_param.state
        # Phase check.
        if nsr.phase != np_state.phase:
            return False

        # Level check.
        if np_state.level < nsr.min_level or np_state.level > nsr.max_level:
            return False

        # Stage check.
        stages = set(np_state.stage)
        for stage in nsr.stage:
            # Missing stage? Exclude.
            if stage not in stages:
                return False
        for not_stage in nsr.not_stage:
            # Stage should be there, but found? Exclude.
            if not_stage in stages:
                return False

        return True

    def _include_layer(self, lp):
        """Check if layer should be included in the network.

        Based on `NetState` of the `NetParameter`.

        Args:
            lp: `LayerParameter` protobuf message.

        Returns:
            Boolean `True` iff layer should be included.
        """
        # Default behavior.
        default = True
        # If inclusion rules defined, exclude by default.
        if lp.include:
            default = False
        # If exclusion rules defined, include by default.
        if lp.exclude:
            default = True

        # Include?
        for include_NSR in lp.include:
            if self._satisfied_NSR(include_NSR):
                return True
        # Exclude?
        for exclude_NSR in lp.exclude:
            if self._satisfied_NSR(exclude_NSR):
                return False
        return default
