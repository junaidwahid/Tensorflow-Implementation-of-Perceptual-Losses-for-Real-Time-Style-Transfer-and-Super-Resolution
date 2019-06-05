import time
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


def test(model,content_img,style_img,output_path):
    sess1 = tf.Session()

    ckpt = tf.train.get_checkpoint_state('./')
    model['saver'].restore(sess1, 'C:/Users/Junaid/PycharmProjects/style_transfer/model/model.ckpt')

    s_image = get_img(style_img, (256, 256, 3))
    images1 = get_img(content_img, (256, 256, 3))
    images1 = images1.reshape((1, 256, 256, 3))
    t = time.time()
    image = sess1.run([model['content_image']],
                      feed_dict={model['x']: images1, model['style_image']: s_image, model['phase']: False,
                                 model['keep_prob']: 1})
    #     print(image[0][0].shape)
    #     return
    cv.imwrite(output_path, image[0][0])
    print("It took", time.time() - t, "seconds .")
    print("Test completed")


if __name__ == "__main__":
    tf.reset_default_graph()

    parser = argparse.ArgumentParser()
    parser.add_argument("-content_img", "--content_img", dest="content_img", default="", help="Image on which style will be apply")
    parser.add_argument("-style_img", "--style_img", dest="style_img", default="", help="style_img")
    parser.add_argument("-output_path", "--output_path", dest="output_path", help="output_path")
    parser.add_argument("-squeezenet_path", "--squeezenet_path", dest="squeezenet_path", help="squeezenet_path")
    args = parser.parse_args()
    content_img,style_img,output_path,squeezenet_path=args.content_img,args.style_img,args.output_path,args.squeezenet_path

    model = build_graph(squeezenet_path,cond='test', batch_size=1)
    test(model,content_img,style_img,output_path)
