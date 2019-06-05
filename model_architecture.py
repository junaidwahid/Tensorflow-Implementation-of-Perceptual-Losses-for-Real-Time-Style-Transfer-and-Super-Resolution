import tensorflow  as tf
from squeezenet import SqueezeNet
from functions import *
from Data_functions import *



def get_session():
    """Create a session that dynamically allocates memory."""

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def model_arch(x, phase, j, keep_prob, batch_size):
    if j == 'restore':
        weight, bias = restore_weights()
    else:
        weight, bias = initialize_weights_2()

    # FIRST CONVOLUTION LAYER
    conv1 = convolution(x, weight['w_conv1'], bias['b_conv1'], [1, 1, 1, 1], 'SAME', "conv1", keep_prob, phase)

    # second CONVOLUTION LAYER
    conv2 = convolution(conv1, weight['w_conv2'], bias['b_conv2'], [1, 2, 2, 1], 'SAME', "conv2", keep_prob, phase)

    # THIRD CONVOLUTION LAYER
    conv3 = convolution(conv2, weight['w_conv3'], bias['b_conv3'], [1, 2, 2, 1], 'SAME', "conv3", keep_prob, phase)

    # FIRST RESIDUAL BLOCK
    identity1 = tf.image.resize_images(conv3, [80, 80])
    res1 = residual_block(conv3, identity1, weight['w_resconv1'], bias['b_resconv1'], weight['w_resconv2'],
                          bias['b_resconv2'], keep_prob, phase)
    # res1=tf.nn.relu(res1)


    # SECOND RESIDUAL BLOCK
    identity2 = tf.image.resize_images(res1, [76, 76])
    res2 = residual_block(res1, identity2, weight['w_resconv3'], bias['b_resconv3'], weight['w_resconv4'],
                          bias['b_resconv4'], keep_prob, phase)
    # res2_relu=tf.nn.relu(res2)



    # THIRD RESIDUAL BLOCK
    identity3 = tf.image.resize_images(res2, [72, 72])
    res3 = residual_block(res2, identity3, weight['w_resconv5'], bias['b_resconv5'], weight['w_resconv6'],
                          bias['b_resconv6'], keep_prob, phase)
    # res3_relu=tf.nn.relu(res3)




    # FOURTH RESIDUAL BLOCK
    identity4 = tf.image.resize_images(res3, [68, 68])
    res4 = residual_block(res3, identity4, weight['w_resconv7'], bias['b_resconv7'], weight['w_resconv8'],
                          bias['b_resconv8'], keep_prob, phase)
    # res4_relu=tf.nn.relu(res4)



    # FIFTH RESIDUAL BLOCK
    identity5 = tf.image.resize_images(res4, [64, 64])
    res5 = residual_block(res4, identity5, weight['w_resconv9'], bias['b_resconv9'], weight['w_resconv10'],
                          bias['b_resconv10'], keep_prob, phase)
    # res5_relu=tf.nn.relu(res5)
    # print(res5.shape) up1=bias['b_up1']+tf.nn.conv2d_transpose(res5,weight['w_up1'],output_shape=[batch_size,128,128,64],strides=[1,2,2,1],padding='SAME')

    # First UPSAMPLE LAYER1
    # First UPSAMPLE LAYER1
    up1 = bias['b_up1'] + tf.nn.conv2d_transpose(res5, weight['w_up1'], output_shape=[batch_size, 128, 128, 64],
                                                 strides=[1, 2, 2, 1], padding='SAME')

    # SECOND UPSAMPLE LAYER2
    up2 = bias['b_up2'] + tf.nn.conv2d_transpose(up1, weight['w_up2'], output_shape=[batch_size, 256, 256, 32],
                                                 strides=[1, 2, 2, 1], padding='SAME')

    # last Convolution LAYER
    content_image = convolution(up2, weight['w_conv4'], bias['b_conv4'], [1, 1, 1, 1], 'SAME', "final_conv", keep_prob,
                                phase)
    content_image = tf.add(content_image, 0, name="final")

    # print(content_image.shape)

    return content_image


def build_graph(squeezenet_path,cond='train', param='initialize', batch_size=1):
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():

            sess = get_session()

        sess1 = tf.Session()
        if cond == 'train':
            SAVE_PATH = squeezenet_path#'C:/Users/Junaid/Desktop/style proj/datasets/squeezenet.ckpt'
            model = SqueezeNet(save_path=SAVE_PATH, sess=sess1)
            # print("kndflkn")

        x = tf.placeholder(tf.float32, [batch_size, 256, 256, 3], name="x")
        phase = tf.placeholder(tf.bool, name='phase')
        style_image = tf.placeholder(tf.float32, shape=[256, 256, 3], name='style_image')
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        padding = [[0, 0], [40, 40], [40, 40], [0, 0]]
        x_new = tf.pad(x, padding, 'REFLECT')

        ########## MODEL ###############
        content_image = model_arch(x_new, phase, param, keep_prob,
                                   batch_size)  # put 1 for intilization and 2 for restoring saved weights in  3rd parametr

        if cond == 'train':
            # parameter initialzation
            content_layer = 3
            content_weight = tf.constant([5e-2])
            style_layers = [1, 4, 6, 7]
            style_weights = [2000.0, 500.0, 12.0, 1.0]
            tv_weight = tf.constant([5e-2])
            # CONTENT_WEIGHT = 7.5e0
            # STYLE_WEIGHT = 1e2
            # TV_WEIGHT = 2e2


            content_original = x

            # calculating_features

            # fake_extract = sess.run(fake_feat[content_layer])
            original_feat = model.extract_features(content_original)
            original_extract = original_feat[content_layer]

            fake_feat = model.extract_features(content_image)
            fake_extract = fake_feat[content_layer]

            style_target = [gram_matrix(original_feat[idx]) for idx in style_layers]
            # fake_style=[fake_feat[idx] for idx in style_layers]





            s_loss = style_loss(fake_feat, style_layers, style_target, style_weights)
            # con_loss==tf.py_func(content_loss,[fake_extract, original_extract],tf.float32)

            con_loss = content_weight * tf.reduce_sum((fake_extract - original_extract) ** 2)

            # tvariation_loss=tv_loss(content_image, tv_weight)#tf.py_func(tv_loss, [content_image, tv_weight], tf.float32)
            w_variance = tf.reduce_sum(tf.pow(content_image[:, :, :-1, :] - content_image[:, :, 1:, :], 2))
            h_variance = tf.reduce_sum(tf.pow(content_image[:, :-1, :, :] - content_image[:, 1:, :, :], 2))
            tvariation_loss = tv_weight * (h_variance + w_variance)
            total_loss = tf.add(s_loss, con_loss, name="total_loss")
            final_loss = tf.add(total_loss, tvariation_loss, name="final_loss")

            ###trainig_initialization

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train = tf.train.AdamOptimizer(1e-2, name="adam1").minimize(total_loss, name="training")

        if cond == 'test':
            return dict(x=x, style_image=style_image, keep_prob=keep_prob, phase=phase, content_image=content_image,
                        saver=tf.train.Saver())
        else:
            return dict(x=x, style_image=style_image, keep_prob=keep_prob, phase=phase, content_image=content_image,
                        final_loss=final_loss, train=train, saver=tf.train.Saver())
