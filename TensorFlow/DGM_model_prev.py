import tensorflow as tf
import numpy as np
from TensorFlow import tf_util

from TensorFlow.layers import DGM
from TensorFlow.layers import GraphConv


def placeholder_inputs(batch_size, num_point, num_features, num_classes):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_classes))

    return pointclouds_pl, labels_pl


def get_model(input_features, num_classes, is_training, param=None, k=5):
    # network parameters
    edgconv_layers = [16, 16]
    fc_layers = [16]  # output size of the fc_layers
    dgm_layers = [16, 16]
    #   k=5
    pooling = tf.reduce_sum

    if not param is None:
        edgconv_layers = param.edgconv_layers
        fc_layers = param.fc_layers
        dgm_layers = param.knnlayers.copy()
        #     k = param.k
        pooling = {'sum': tf.reduce_sum, 'max': tf.reduce_max}[param.pooling]

    # input shape
    input_features = tf.expand_dims(input_features, -2)

    dims = input_features.get_shape().as_list()
    dims[-1] = 0
    X_g = tf.zeros(shape=dims, dtype=tf.float32)  # features designated to the graph representation learning
    X = input_features  # features designated to the node representation learning
    edges = None

    nets = []
    ps = []
    ed = []
    for i, (dgm_l, edg_l) in enumerate(zip(dgm_layers, edgconv_layers)):

        ## Graph learning branch ##
        conv_func = lambda x, e: x
        if dgm_l > 0:
            # define the graph feature learning function f_theta
            conv_func = lambda x, e: GraphConv(x, e, dgm_l, k, scope='dgm_fn_layer_%d' % i, is_training=is_training,
                                               dropout=0.1)
        #         conv_func = lambda x, e : MLP(x, e, [dgm_l,dgm_l], k, scope='dgm_fn_layer_%d' % i, is_training=is_training, dropout=0)
        #         conv_func = lambda x, e : tf.identity(x)

        # DGM module which takes as input features 'X_g', set of edges 'E', number of neighbors k, f_theta.
        # DGM module outputs graph representation learning features X_g, set of edges 'E', and the
        # probability of the edges for l+1th layer.
        X_g, edges, probs = DGM(tf.concat([X_g, tf.stop_gradient(X)], -1), edges, k, conv_func, is_training=is_training,
                                scope='dgm_layer_%d' % i)
        #     X_g,edges,probs = DGM(tf.stop_gradient(X), edges, k, conv_func, is_training=is_training, scope='dgm_layer_%d' % i)

        ## Node learning branch ##
        # please refer to figure 2 in the paper,
        #     X = EdgeConv(X, edges, edg_l, k, scope='conv_layer_%d' % i, is_training=is_training, dropout=0.1) # GraphConv block in figure 2.
        X = GraphConv(X, edges, edg_l, k, scope='conv_layer_%d' % i, is_training=is_training,
                      dropout=0.1)  # GraphConv block in figure 2.

        ps.append(probs)
        ed.append(edges)

    # finally FC layers to get the prediction
    net = X
    for i, fc in enumerate(fc_layers):
        net = tf_util.conv2d(net, fc, [1, 1], padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training, scope='dp%d' % i, is_dist=True)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp%d' % i)

    net = tf_util.conv2d(net, num_classes, [1, 1], padding='VALID', stride=[1, 1],
                         activation_fn=None, scope='seg/conv3', is_dist=True)
    net = tf.squeeze(net, [2])

    P = tf.stack(ps)
    E = tf.stack(ed)
    return net, P, E


def masked_softmax_cross_entropy(preds, labels, mask, num_classes):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)

    loss *= mask

    return loss


def get_loss(pred, label, mask, P):
    """ pred: B,N,13; label: B,N """
    num_classes = pred.shape[-1]
    loss = masked_softmax_cross_entropy(pred, label, mask, num_classes)

    # per class accuracy
    hard_res = tf.argmax(pred, 2)
    corr_res = tf.argmax(label, 2)
    corr_pred = tf.cast(tf.equal(hard_res, corr_res), tf.float32)
    wron_pred = 1 - corr_pred
    P = tf.reduce_sum(tf.log(P + 1e-10), [0, 3])

    unique_idx, unique_back = tf.unique(tf.reshape(corr_res, (-1,)))
    # class_mask = tf.cast(label,tf.float32)
    ten_ = np.ones([1, 14503, 1])
    class_mask = tf.cast(ten_, tf.float32)
    class_mask = class_mask * mask[None, :, None]

    per_class_acc = tf.reduce_sum(corr_pred[..., None] * class_mask, [0, -2]) / tf.reduce_sum(class_mask, [0, -2])

    perpoint_weight = tf.gather(per_class_acc, unique_back)
    perpoint_weight = tf.reshape(perpoint_weight, (corr_res.shape[0], -1))

    samp_loss = (-1 * tf.reduce_sum(corr_pred * P * (1 - perpoint_weight) * mask) + \
                 1 * tf.reduce_sum(wron_pred * P * perpoint_weight * mask)) / tf.reduce_sum(mask)

    #   samp_loss = tf.identity(0.0)
    samp_loss = samp_loss
    return samp_loss, tf.reduce_mean(loss)
