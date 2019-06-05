from functions import *
from model_arch import *
from Data_functions import *
import argparse


def get_session():
    """Create a session that dynamically allocates memory."""

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def train(model, num_epoch, num_train, batch_size, param, model_path, style_img, dataset_path):
    sess1 = tf.Session()
    if param == 'restore':
        ckpt = tf.train.get_checkpoint_state('./')
        model['saver'].restore(sess1, model_path)
    else:
        sess1.run(tf.global_variables_initializer())

    batches_list = list(range(int(ceil(float(num_train) / batch_size))))

    s_image = get_img(style_img, (256, 256, 3))  # ('C:/Users/Junaid/Desktop/style proj/style_image.jpg', (256, 256, 3))

    hdf5_file = hdf.File(dataset_path)  # ('C:/Users/Junaid/Desktop/style proj/datasets/dataset_git.hdf5', "r")

    for ep in range(num_epoch):
        shuffle(batches_list)
        for n, i in enumerate(batches_list):
            i_s = i * batch_size  # index of the first image in this batch
            i_e = min([(i + 1) * batch_size, num_train])  # index of the last image in this batch
            # read batch images
            images1 = hdf5_file["train_img"][i_s:i_e, ...]

            loss, _ = sess1.run([model['final_loss'], model['train']],
                                feed_dict={model['x']: images1, model['style_image']: s_image, model['phase']: True,
                                           model['keep_prob']: 0.5})

            if n % 5 == 0:
                loss, _ = sess1.run([model['final_loss'], model['train']],
                                    feed_dict={model['x']: images1, model['style_image']: s_image, model['phase']: True,
                                               model['keep_prob']: 0.5})
                print("loss is %d", loss)
            else:
                sess1.run([model['train']],
                          feed_dict={model['x']: images1, model['style_image']: s_image, model['phase']: True,
                                     model['keep_prob']: 0.5})

    model['saver'].save(sess1, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -db DATABASE -u USERNAME -p PASSWORD -size 20000
    parser.add_argument("-param", "--param", dest="param", default="init",
                        help="wheather you want to train your model from scratch or restored it from somewhere")
    parser.add_argument("-num_epoch", "--num_epoch", dest="num_epoch", default="16", help="Total number of epoch")
    parser.add_argument("-model_path", "--model_path", dest="model_path",
                        help="The path where you want to save your model")
    parser.add_argument("-train_size", "--train_size", dest="train_size", default="16", help="dataset size")
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default="16", help="Batch size")
    parser.add_argument("-style_img", "--style_img", dest="style_img", help="The path of style image")
    parser.add_argument("-dataset_path", "--dataset_path", dest="dataset_path", help="The dataset(hdf5 file) path")
    parser.add_argument("-squeezenet_path", "--squeezenet_path", dest="squeezenet_path", help="The squeezenet_path path")
    args = parser.parse_args()
    num_epoch, param, num_train, batch_size, model_path, style_img, dataset_path,squeezenet_path = args.num_epoch, args.param, args.train_size, args.batch_size, args.model_path, args.style_img, args.dataset_path,args.squeezenet_path
    print(num_epoch, param, num_train, batch_size, model_path, style_img, dataset_path)

    tf.reset_default_graph()
    model = build_graph(squeezenet_path,cond='train', param="init", batch_size=1)  ## change initialize to restore
    train(model, int(num_epoch), int(num_train), int(batch_size), param, model_path, style_img, dataset_path)
