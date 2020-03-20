"""
model_pwcnet.py

PWC-Net model class.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import pathlib
import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

_DEFAULT_PWCNET_TEST_OPTIONS = {
    'ckpt_path': 'pwc_models/sintel_gray_weights/pwcnet.sintel_gray.ckpt-54000',
    'controller': '/device:CPU:0',
    'batch_size': 1,
    'pyr_lvls': 6, 
    'flow_pred_lvl': 2,  
    'search_range': 4
}

# from ref_model import PWCNet

def cost_volume(c1, warp, search_range, name):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return cost_vol

def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = array_ops.unstack(array_ops.shape(grid))
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received: '
            raise ValueError(msg + str(shape))

        batch_size, height, width, channels = shape
        query_type = query_points.dtype
        query_shape = array_ops.unstack(array_ops.shape(query_points))
        grid_type = grid.dtype

        if len(query_shape) != 3:
            msg = ('Query points must be 3 dimensional. Received: ')
            raise ValueError(msg + str(query_shape))

        _, num_queries, _ = query_shape

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        flattened_grid = array_ops.reshape(grid,
                                           [batch_size * height * width, channels])
        batch_offsets = array_ops.reshape(
            math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp

def dense_image_warp(image, flow, name='dense_image_warp'):
    with ops.name_scope(name):
        batch_size, height, width, channels = array_ops.unstack(array_ops.shape(image))
        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = array_ops.meshgrid(
            math_ops.range(width), math_ops.range(height))
        stacked_grid = math_ops.cast(
            array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
        query_points_on_grid = batched_grid - flow
        query_points_flattened = array_ops.reshape(query_points_on_grid,
                                                   [batch_size, height * width, 2])
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = _interpolate_bilinear(image, query_points_flattened)
        interpolated = array_ops.reshape(interpolated,
                                         [batch_size, height, width, channels])
        return interpolated

class ModelPWCNet:
    def __init__(self, name='pwcnet', session=None, options=_DEFAULT_PWCNET_TEST_OPTIONS):
        self.opts = options
        self.y_hat_train_tnsr = self.y_hat_val_tnsr = self.y_hat_test_tnsr = None
        self.name = name

        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Configure a TF session, if one doesn't already exist
            if session is None:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.allow_soft_placement = True
                self.sess = tf.Session(config=config)
            else:
                self.sess = session

            # Build the TF graph
            batch_size = self.opts['batch_size']
            self.x_tnsr = tf.placeholder(tf.float32, [batch_size] + [2, None, None, 3], 'x_tnsr')
            self.y_tnsr = tf.placeholder(tf.float32, [batch_size] + [None, None, 2], 'y_tnsr')
            
            # Build the backbone neural nets and collect the output tensors
            with tf.device(self.opts['controller']):
                self.flow_pred_tnsr, self.flow_pyr_tnsr = self.nn(self.x_tnsr)

            # Set output tensors
            self.y_hat_test_tnsr = [self.flow_pred_tnsr, self.flow_pyr_tnsr]

            # Init saver (override if you wish) and load checkpoint if it exists
            self.saver = tf.train.Saver()
            
            # Initialize the graph with the content of the checkpoint
            self.last_ckpt = os.path.join(os.getcwd(), self.opts['ckpt_path'])
            assert(self.last_ckpt is not None)
            self.saver.restore(self.sess, self.last_ckpt)

    ###
    # Sample mgmt
    ###
    def adapt_x(self, x):
        # Ensure we're dealing with RGB image pairs
        assert (isinstance(x, np.ndarray) or isinstance(x, list))
        if isinstance(x, np.ndarray):
            assert (len(x.shape) == 5)
            assert (x.shape[1] == 2 and x.shape[4] == 3)
        else:
            assert (len(x[0].shape) == 4)
            assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

        # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
        x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
        x_adapt /= 255.

        # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
        _, pad_h = divmod(x_adapt.shape[2], 2**self.opts['pyr_lvls'])
        if pad_h != 0:
            pad_h = 2 ** self.opts['pyr_lvls'] - pad_h
        _, pad_w = divmod(x_adapt.shape[3], 2**self.opts['pyr_lvls'])
        if pad_w != 0:
            pad_w = 2 ** self.opts['pyr_lvls'] - pad_w
        x_adapt_info = None
        if pad_h != 0 or pad_w != 0:
            padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
            x_adapt_info = x_adapt.shape  # Save original shape
            x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

        return x_adapt, x_adapt_info

    def postproc_y_hat_test(self, y_hat, adapt_info=None):
        assert (isinstance(y_hat, list) and len(y_hat) == 2)

        # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
        pred_flows = y_hat[0]
        if adapt_info is not None:
            pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]

        # Individuate flows of the flow pyramid (at this point, they are still batched)
        pyramids = y_hat[1]
        pred_flows_pyramid = []
        for idx in range(len(pred_flows)):
            pyramid = []
            for lvl in range(self.opts['pyr_lvls'] - self.opts['flow_pred_lvl'] + 1):
                pyramid.append(pyramids[lvl][idx])
            pred_flows_pyramid.append(pyramid)

        return pred_flows, pred_flows_pyramid

    def predict_from_img_pairs(self, img_pairs, batch_size=1):
        with self.graph.as_default():
            # Chunk image pair list
            batch_size = self.opts['batch_size']
            test_size = len(img_pairs)
            rounds, rounds_left = divmod(test_size, batch_size)
            if rounds_left:
                rounds += 1

            # Loop through input samples and run inference on them
            preds, test_ptr = [], 0
            for _round in range(rounds):
                # In batch mode, make sure to wrap around if there aren't enough input samples to process
                if test_ptr + batch_size < test_size:
                    new_ptr = test_ptr + batch_size
                    indices = list(range(test_ptr, test_ptr + batch_size))
                else:
                    new_ptr = (test_ptr + batch_size) % test_size
                    indices = list(range(test_ptr, test_size)) + list(range(0, new_ptr))
                test_ptr = new_ptr

                # Repackage input image pairs as np.ndarray
                x = np.array([img_pairs[idx] for idx in indices])

                # Make input samples conform to the network's requirements
                # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
                x_adapt, x_adapt_info = self.adapt_x(x)
                if x_adapt_info is not None:
                    y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
                else:
                    y_adapt_info = None

                # Run the adapted samples through the network
                feed_dict = {self.x_tnsr: x_adapt}
                y_hat = self.sess.run(self.y_hat_test_tnsr, feed_dict=feed_dict)
                y_hats, _ = self.postproc_y_hat_test(y_hat, y_adapt_info)

                # Return flat list of predicted labels
                for y_hat in y_hats:
                    preds.append(y_hat)

        return preds[0:test_size]

    ###
    # PWC-Net pyramid helpers
    ###
    def extract_features(self, x_tnsr, name='featpyr'):
        assert(1 <= self.opts['pyr_lvls'] <= 6)
        # Make the feature pyramids 1-based for better readability down the line
        num_chann = [None, 16, 32, 64, 96, 128, 196]
        c1, c2 = [None], [None]
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
                for lvl in range(1, self.opts['pyr_lvls'] + 1):
                    # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                    # reuse is set to True because we want to learn a single set of weights for the pyramid
                    # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                    f = num_chann[lvl]
                    x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name='conv{}a'.format(lvl), reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1)  
                    x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv{}aa'.format(lvl), reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1) 
                    x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name='conv{}b'.format(lvl), reuse=reuse)
                    x = tf.nn.leaky_relu(x, alpha=0.1, name=str(name)+str(lvl))
                    pyr.append(x)
        return c1, c2

    ###
    # PWC-Net warping helpers
    ###
    def warp(self, c2, sc_up_flow, lvl, name='warp'):
        op_name = str(name)+str(lvl)
        with tf.name_scope(name):
            return dense_image_warp(c2, sc_up_flow, name=op_name)

    def deconv(self, x, lvl, name='up_flow'):
        op_name = str(name)+str(lvl)
        with tf.variable_scope('upsample'):
            # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
            return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)

    ###
    # Cost Volume helpers
    ###
    def corr(self, c1, warp, lvl, name='corr'):
        op_name = 'corr'+str(lvl)
        with tf.name_scope(name):
            return cost_volume(c1, warp, self.opts['search_range'], op_name)

    ###
    # Optical flow estimator helpers
    ###
    def predict_flow(self, corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
        op_name = 'flow'+str(lvl)
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            if c1 is None and up_flow is None and up_feat is None:
                x = corr
            else:
                x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

            conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv{}_0'.format(lvl))
            act = tf.nn.leaky_relu(conv, alpha=0.1)  # default alpha is 0.2 for TF
            x = tf.concat([act, x], axis=3) 

            conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv{}_1'.format(lvl))
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) 

            conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name='conv{}_2'.format(lvl))
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) 

            conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name='conv{}_3'.format(lvl))
            act = tf.nn.leaky_relu(conv, alpha=0.1)
            x = tf.concat([act, x], axis=3) 

            conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name='conv{}_4'.format(lvl))
            act = tf.nn.leaky_relu(conv, alpha=0.1)  # will also be used as an input by the context network
            upfeat = tf.concat([act, x], axis=3, name='upfeat'+str(lvl)) 

            flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

            return upfeat, flow

    ###
    # PWC-Net context network helpers
    ###
    def refine_flow(self, feat, flow, lvl, name='ctxt'):
        op_name = 'refined_flow'+str(lvl)
        init = tf.keras.initializers.he_normal()
        with tf.variable_scope(name):
            x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}1'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)  # default alpha is 0.2 for TF
            x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name='dc_conv{}2'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name='dc_conv{}3'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name='dc_conv{}4'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name='dc_conv{}5'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}6'.format(lvl))
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='dc_conv{}7'.format(lvl))

            return tf.add(flow, x, name=op_name)

    ###
    # PWC-Net nn builder
    ###
    def nn(self, x_tnsr, name='pwcnet'):
        with tf.variable_scope(name):

            # Extract pyramids of CNN features from both input images (1-based lists))
            c1, c2 = self.extract_features(x_tnsr)

            flow_pyr = []

            for lvl in range(self.opts['pyr_lvls'], self.opts['flow_pred_lvl'] - 1, -1):

                if lvl == self.opts['pyr_lvls']:
                    # Compute the cost volume
                    corr = self.corr(c1[lvl], c2[lvl], lvl)

                    # Estimate the optical flow
                    upfeat, flow = self.predict_flow(corr, None, None, None, lvl)
                else:
                    # Warp level of Image1's using the upsampled flow
                    scaler = 20. / 2**lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                    warp = self.warp(c2[lvl], up_flow * scaler, lvl)

                    # Compute the cost volume
                    corr = self.corr(c1[lvl], warp, lvl)

                    # Estimate the optical flow
                    upfeat, flow = self.predict_flow(corr, c1[lvl], up_flow, up_feat, lvl)

                _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

                if lvl != self.opts['flow_pred_lvl']:
                    flow = self.refine_flow(upfeat, flow, lvl)

                    # Upsample predicted flow and the features used to compute predicted flow
                    flow_pyr.append(flow)

                    up_flow = self.deconv(flow, lvl, 'up_flow')
                    up_feat = self.deconv(upfeat, lvl, 'up_feat')
                else:
                    # Refine the final predicted flow
                    flow = self.refine_flow(upfeat, flow, lvl)
                    flow_pyr.append(flow)

                    # Upsample the predicted flow (final output) to match the size of the images
                    scaler = 2**self.opts['flow_pred_lvl']
                    size = (lvl_height * scaler, lvl_width * scaler)
                    flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scaler
                    break

            return flow_pred, flow_pyr
