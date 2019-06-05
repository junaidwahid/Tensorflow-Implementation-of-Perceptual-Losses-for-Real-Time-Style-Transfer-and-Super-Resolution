
import tensorflow as tf


def get_session():
    """Create a session that dynamically allocates memory."""

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def gram_matrix(features, normalize=True):
    shapes = tf.shape(features)
    feat_reshape = tf.reshape(features, [1, -1, shapes[3]])
    gram = tf.matmul(tf.transpose(feat_reshape, perm=[0, 2, 1]), tf.transpose(feat_reshape, perm=[0, 1, 2]))
    n_neurons = tf.cast(shapes[1] * shapes[2] * shapes[3], tf.float32)
    gram_norm = gram / n_neurons
    return gram_norm


def style_loss(feats, style_layers, style_targets, style_weights):
    style_loss1 = tf.constant(0.0, tf.float32)
    for i in range(len(style_layers)):
        gram = gram_matrix(feats[style_layers[i]])
        style_loss1 += style_weights[i] * tf.reduce_sum(tf.pow(gram - style_targets[i], 2))
    return style_loss1


# tf.reset_default_graph()
def initialize_weigths(shape, name):
    # initial=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    # return tf.Variable(initial,dtype=tf.float32,name=name)


def initialize_bias(shape, name):
    # initial=tf.constant(0.1,shape=shape)
    return tf.get_variable(name, initializer=tf.constant(0.1, shape=shape))
    # return tf.Variable(initial,name=name)


def conv2d(x, w, stride, pad, name):
    return tf.nn.conv2d(x, w, strides=stride, padding=pad, name=name)


def pooling(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], name=name)


# tf.reset_default_graph()

# CONVOLUTION FUNCTION:



# RESIDUAL BLOCK FUNCTION
def residual_block(x, identity, w_resconv1, b_resconv1, w_resconv2, b_resconv2, keep_prob, phase):
    resconv1 = conv2d(x, w_resconv1, [1, 1, 1, 1], 'VALID', "resconv1")
    new_resconv1 = resconv1 + b_resconv1
    res_batch1 = tf.contrib.layers.batch_norm(new_resconv1, center=True, scale=True, is_training=phase)
    res1_relu = tf.nn.relu(res_batch1)
    res_drop1 = tf.nn.dropout(res1_relu, keep_prob)

    resconv2 = conv2d(res_drop1, w_resconv2, [1, 1, 1, 1], 'VALID', "resconv2")
    new_resconv2 = resconv2 + b_resconv2
    res_batch2 = tf.contrib.layers.batch_norm(new_resconv2, center=True, scale=True, is_training=phase)
    res2_drop = tf.nn.dropout(res_batch2, keep_prob)
    mapping = res2_drop + identity
    return mapping


def convolution(x, w_conv1, b_conv1, stride, pad, name, keep_prob, phase):
    last_name = "relu_" + name
    conv1 = conv2d(x, w_conv1, stride, pad, name)
    new_conv1 = conv1 + b_conv1
    batch1 = tf.contrib.layers.batch_norm(new_conv1, center=True, scale=True, is_training=phase)
    conv1_relu = tf.nn.relu(batch1, name=last_name)
    return tf.nn.dropout(conv1_relu, keep_prob)


def initialize_weights_2():
    w_conv1 = initialize_weigths([9, 9, 3, 32], "w_conv1")
    b_conv1 = initialize_bias([32], "b_conv1")

    w_conv2 = initialize_weigths([3, 3, 32, 64], "w_conv2")
    b_conv2 = initialize_bias([64], "b_conv2")

    # THIRD CONVOLUTION LAYER
    w_conv3 = initialize_weigths([3, 3, 64, 128], "w_conv3")
    b_conv3 = initialize_bias([128], "b_conv3")

    ##################RESNET WEIGHTS#######################
    w_resconv1 = initialize_weigths([3, 3, 128, 128], name="w_resconv1")
    b_resconv1 = initialize_bias([128], name="b_resconv1")
    w_resconv2 = initialize_weigths([3, 3, 128, 128], name="w_resconv2")
    b_resconv2 = initialize_bias([128], name="b_resconv2")

    w_resconv3 = initialize_weigths([3, 3, 128, 128], name="w_resconv3")
    b_resconv3 = initialize_bias([128], name="b_resconv3")
    w_resconv4 = initialize_weigths([3, 3, 128, 128], name="w_resconv4")
    b_resconv4 = initialize_bias([128], name="b_resconv4")

    w_resconv5 = initialize_weigths([3, 3, 128, 128], name="w_resconv5")
    b_resconv5 = initialize_bias([128], name="b_resconv5")
    w_resconv6 = initialize_weigths([3, 3, 128, 128], name="w_resconv6")
    b_resconv6 = initialize_bias([128], name="b_resconv6")

    w_resconv7 = initialize_weigths([3, 3, 128, 128], name="w_resconv7")
    b_resconv7 = initialize_bias([128], name="b_resconv7")
    w_resconv8 = initialize_weigths([3, 3, 128, 128], name="w_resconv8")
    b_resconv8 = initialize_bias([128], name="b_resconv8")

    w_resconv9 = initialize_weigths([3, 3, 128, 128], name="w_resconv9")
    b_resconv9 = initialize_bias([128], name="b_resconv9")
    w_resconv10 = initialize_weigths([3, 3, 128, 128], name="w_resconv10")
    b_resconv10 = initialize_bias([128], name="b_resconv10")

    #############tranpose layers###############
    w_up1 = initialize_weigths([3, 3, 64, 128], "w_up1")
    b_up1 = initialize_bias([64], "b_up1")

    # SECOND UPSAMPLE LAYER
    w_up2 = initialize_weigths([3, 3, 32, 64], "w_up2")
    b_up2 = initialize_bias([32], "b_up2")

    # last Convolution LAYER
    w_conv4 = initialize_weigths([9, 9, 32, 3], "w_conv4")
    b_conv4 = initialize_bias([3], "b_conv4")

    weight = {"w_conv1": w_conv1, "w_conv2": w_conv2, "w_conv3": w_conv3, "w_conv4": w_conv4, "w_up1": w_up1,
              "w_up2": w_up2,
              "w_resconv1": w_resconv1, "w_resconv2": w_resconv2, "w_resconv3": w_resconv3, "w_resconv4": w_resconv4,
              "w_resconv5": w_resconv5,
              "w_resconv6": w_resconv6, "w_resconv7": w_resconv7, "w_resconv8": w_resconv8, "w_resconv9": w_resconv9,
              "w_resconv10": w_resconv10}

    bias = {"b_conv1": b_conv1, "b_conv2": b_conv2, "b_conv3": b_conv3, "b_conv4": b_conv4, "b_up1": b_up1,
            "b_up2": b_up2,
            "b_resconv1": b_resconv1, "b_resconv2": b_resconv2, "b_resconv3": b_resconv3, "b_resconv4": b_resconv4,
            "b_resconv5": b_resconv5,
            "b_resconv6": b_resconv6, "b_resconv7": b_resconv7, "b_resconv8": b_resconv8, "b_resconv9": b_resconv9,
            "b_resconv10": b_resconv10}

    return weight, bias


def restore_weights():
    w_conv1 = tf.get_variable("w_conv1", [9, 9, 3, 32])
    b_conv1 = tf.get_variable("b_conv1", [32])

    w_conv2 = tf.get_variable("w_conv2", [3, 3, 32, 64])
    b_conv2 = tf.get_variable("b_conv2", [64])

    # THIRD CONVOLUTION LAYER
    w_conv3 = tf.get_variable("w_conv3", [3, 3, 64, 128])
    b_conv3 = tf.get_variable("b_conv3", [128])

    ##################RESNET WEIGHTS#######################
    w_resconv1 = tf.get_variable("w_resconv1", [3, 3, 128, 128])
    b_resconv1 = tf.get_variable("b_resconv1", [128])
    w_resconv2 = tf.get_variable("w_resconv2", [3, 3, 128, 128])
    b_resconv2 = tf.get_variable("b_resconv2", [128])

    w_resconv3 = tf.get_variable("w_resconv3", [3, 3, 128, 128])
    b_resconv3 = tf.get_variable("b_resconv3", [128])
    w_resconv4 = tf.get_variable("w_resconv4", [3, 3, 128, 128])
    b_resconv4 = tf.get_variable("b_resconv4", [128])

    w_resconv5 = tf.get_variable("w_resconv5", [3, 3, 128, 128])
    b_resconv5 = tf.get_variable("b_resconv5", [128])
    w_resconv6 = tf.get_variable("w_resconv6", [3, 3, 128, 128])
    b_resconv6 = tf.get_variable("b_resconv6", [128])

    w_resconv7 = tf.get_variable("w_resconv7", [3, 3, 128, 128])
    b_resconv7 = tf.get_variable("b_resconv7", [128])
    w_resconv8 = tf.get_variable("w_resconv8", [3, 3, 128, 128])
    b_resconv8 = tf.get_variable("b_resconv8", [128])

    w_resconv9 = tf.get_variable("w_resconv9", [3, 3, 128, 128])
    b_resconv9 = tf.get_variable("b_resconv9", [128])
    w_resconv10 = tf.get_variable("w_resconv10", [3, 3, 128, 128])
    b_resconv10 = tf.get_variable("b_resconv10", [128])

    #############tranpose layers###############
    w_up1 = tf.get_variable("w_up1", [3, 3, 64, 128])
    b_up1 = tf.get_variable("b_up1", [64])

    # SECOND UPSAMPLE LAYER
    w_up2 = tf.get_variable("w_up2", [3, 3, 32, 64])
    b_up2 = tf.get_variable("b_up2", [32])

    # last Convolution LAYER
    w_conv4 = tf.get_variable("w_conv4", [9, 9, 32, 3])
    b_conv4 = tf.get_variable("b_conv4", [3])

    weight = {"w_conv1": w_conv1, "w_conv2": w_conv2, "w_conv3": w_conv3, "w_conv4": w_conv4, "w_up1": w_up1,
              "w_up2": w_up2,
              "w_resconv1": w_resconv1, "w_resconv2": w_resconv2, "w_resconv3": w_resconv3, "w_resconv4": w_resconv4,
              "w_resconv5": w_resconv5,
              "w_resconv6": w_resconv6, "w_resconv7": w_resconv7, "w_resconv8": w_resconv8, "w_resconv9": w_resconv9,
              "w_resconv10": w_resconv10}

    bias = {"b_conv1": b_conv1, "b_conv2": b_conv2, "b_conv3": b_conv3, "b_conv4": b_conv4, "b_up1": b_up1,
            "b_up2": b_up2,
            "b_resconv1": b_resconv1, "b_resconv2": b_resconv2, "b_resconv3": b_resconv3, "b_resconv4": b_resconv4,
            "b_resconv5": b_resconv5,
            "b_resconv6": b_resconv6, "b_resconv7": b_resconv7, "b_resconv8": b_resconv8, "b_resconv9": b_resconv9,
            "b_resconv10": b_resconv10}

    return weight, bias
