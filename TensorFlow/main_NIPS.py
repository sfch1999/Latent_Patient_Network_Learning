# import matplotlib.pyplot as plt
import pickle
from TensorFlow.DGM_model import *

for k in [5]:
    print('EXPERIMENTS FOR K=%d' % k)
    tf.reset_default_graph()

    rep = 9 * 0 + 1  # replicate dataset nodes rep times
    batch_size = 1  # Since the whole population is incorporated in one graph.

    #     with open('../UKBB.pickle', 'rb') as f:
    with open('train_data.pickle', 'rb') as f:
        X_, y_, train_mask_, test_mask_, weight_ = pickle.load(f)  # Load the data

    # print(X_)
    # print(y_)

    X_ = X_[..., :10]  # For DGM we use modality 1 (M1) for both node representation and graph learning.
    y_ = y_[None, ...]  # Labels

    X_ = np.tile(X_, [rep, 1, 1])
    y_ = np.tile(y_, [1, rep, 1, 1])
    test_mask_ = np.tile(test_mask_, [rep, 1])
    train_mask_ = np.tile(train_mask_, [rep, 1])

    num_point = X_.shape[0]  # Number of samples in the dataset
    num_features = X_.shape[-2]  # number of features per sample
    num_classes = y_.shape[-2]  # Classes : Normal (N), Mild Cognitive Impairment (MCI) and Alzheimer's (AD)
    # print(num_point,num_features,num_classes)

    is_training = tf.placeholder_with_default(True, shape=())  # to enable the training phase
    pl_mask = tf.placeholder(tf.float32, shape=(
        num_point,))  # To mask out the training and testing samples in transductive setting for during training and testing respectively.

    pl_X, pl_y = placeholder_inputs(batch_size, num_point, num_features, num_classes)  # placeholders for training data

    pred, P, lat = get_model(pl_X, num_classes, is_training,
                             k=k)  # Model outputs prediction and sampled probabilistic graph

    graph_loss, node_loss = get_loss(pred, pl_y, pl_mask,
                                     P)  # L_graph as graph loss and node_loss is the Categorical Cross-Entropy

    loss = graph_loss + node_loss

    acc = tf.reduce_sum(
        tf.cast(tf.equal(tf.argmax(pred, 2), tf.argmax(pl_y, 2)), tf.float32) * pl_mask) / tf.reduce_sum(
        pl_mask)  # accuracy

    lr = tf.placeholder_with_default(1e-3, shape=None)

    opt = tf.train.AdamOptimizer(learning_rate=lr)  # Create an optimizer with the desired parameters.
    opt_op1 = opt.minimize(loss)

    # Intialize the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    folds_accuracy = []
    for fold in range(10):
        # getting the data of the respective fold
        X = np.expand_dims(X_[:, :num_features, fold], axis=0)
        y = y_[:, :, :, fold]
        train_mask = train_mask_[:, fold]
        test_mask = test_mask_[:, fold]
        weight = np.squeeze(weight_[:, fold])

        # Re-initialize session
        tf.set_random_seed(100)
        sess.run(tf.global_variables_initializer())

        # TRAIN with train_mask
        init_lr = 1e-1  # learning rate
        feed_dict = dict({pl_X: X, pl_y: y, pl_mask: train_mask, is_training: True})
        # Traning for 300 epochs
        t = time.time()
        for i in range(301):
            if i % 100 == 0:
                init_lr /= 5
                feed_dict.update({lr: init_lr})

            _, l1, l2, a, pp, l = sess.run([opt_op1, graph_loss, node_loss, acc, P, lat], feed_dict=feed_dict)
        #             _ = sess.run([opt_op1], feed_dict=feed_dict)
        #             if i % 3 == 0:
        #                 print('Iter %d] Graph loss: %.2e  Task loss: %.2e,  Acc:%.1f' % (i, l1, l2, a * 100))
        #         print('Time per iteration: %.2e' % ((time.time()-t)/10))

        # Testing
        feed_dict = dict({pl_X: X, pl_y: y, pl_mask: test_mask, is_training: False})
        l1, l2, a, p, pp, l = sess.run([graph_loss, node_loss, acc, pred, P, lat], feed_dict=feed_dict)
        t = time.time()
        for i in range(7):
            p_ = sess.run(pred, feed_dict=feed_dict)
            p += p_
        #         print('TEST Time per iteration: %.2e' % ((time.time()-t)/70))
        ac = np.sum((np.argmax(p, -1) == np.argmax(y, -1)) * test_mask) / np.sum(test_mask)

        #         print('TEST       : Graph loss: %.2e  Task loss: %.2e,  Acc: %.1f,  Acc2: %.1f' % (l1, l2, a * 100, ac*100))
        #         aaa
        folds_accuracy.append(ac)

    folds_accuracy = np.asarray(folds_accuracy)
    np.savetxt('accuracy_%d.txt' % k, folds_accuracy, delimiter=',')
    print('AVERAGE ACCURACY: %.2f (%.2f)' % (np.mean(folds_accuracy) * 100, np.std(folds_accuracy) * 100))
