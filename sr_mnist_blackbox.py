from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from src.cleverhans.cleverhans.utils_mnist import data_mnist
from importCifar10 import data_cifar10
from src.cleverhans.cleverhans.utils import to_categorical
from src.cleverhans.cleverhans.utils import set_log_level
from src.cleverhans.cleverhans.utils_tf import model_train, model_eval, batch_eval
from src.cleverhans.cleverhans.attacks import FastGradientMethod
from src.cleverhans.cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax

from matplotlib import pyplot as plt
import os
import time
import cv2

FLAGS = flags.FLAGS


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True


def prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate,
              rng):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """
# shape=(None, int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3]))
    # Define TF model graph (for the black-box model)
    print("Before make_basic_cnn")
    model = make_basic_cnn(input_shape=(None, int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3])))
    print('make_basic_cnn() done')
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, predictions, X_train, Y_train, verbose=False,
                args=train_params, rng=rng)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy


def substitute_model(img_rows=28, img_cols=28, num_channels = 1, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, num_channels)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    print("INSIDE train_sub - ")
    print("SHape of x- " + str(x.shape))
    print("SHape of y- " + str(y.shape))

    # x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    # y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]))
    model_sub = substitute_model(img_rows=int(x.shape[1]), img_cols=int(x.shape[2]), num_channels = int(x.shape[3]), nb_classes= int(y.shape[1]))
    print(type(model_sub))
    print(model_sub)
    preds_sub = model_sub(x)
    print(type(preds_sub))
    print(preds_sub)
    print("Defined TensorFlow model graph for the subtstitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    init_all=False, verbose=False, args=train_params,
                    rng=rng)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub

def knowLimits(a, nm=""):
    print("Limits- ", nm, ":-", str(a.min()), str(a.max()))

def mnist_blackbox(cntdill, train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session
    sess = tf.Session()

    # Get MNIST data
    # dataset = 'mnist'
    # X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                               train_end=train_end,
    #                                               test_start=test_start,
    #                                               test_end=test_end)

    # Get CIFAR-10 data
    dataset = 'cifar10_255'
    # X_train, Y_train, X_test, Y_test = data_cifar10()
    # X_train, Y_train, X_test, Y_test = np.float32(X_train), np.float32(Y_train), np.float32(X_test), np.float32(Y_test)

    img_X_train, Y_train, img_X_test, Y_test = data_cifar10()
    img_X_train, Y_train, img_X_test, Y_test = np.float32(img_X_train), np.float32(Y_train), np.float32(img_X_test), np.float32(Y_test)
    # img = cv2.imread('fig.png')
    # print(img_X_train.shape)
    X_train = []
    uChannel_train = []
    vChannel_train = []
    X_test = []
    uChannel_test = []
    vChannel_test = []

    for img in img_X_train:
        # img = img.astype(np.uint8) # it is float32
        # print(img.dtype, img.shape)
        # print(img.min(), img.max())
        # cmp_rgb(img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        X_train.append(y)   # y channel (grayscale)
        uChannel_train.append(u)
        vChannel_train.append(v)
        # print(img)
        # break

    for img in img_X_test:
        # img = img.astype(np.uint8) # it is float32
        # print(img.dtype, img.shape)
        # print(img.min(), img.max())
        # cmp_rgb(img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(img_yuv)
        X_test.append(y)   # y channel (grayscale)
        uChannel_test.append(u)
        vChannel_test.append(v)
        # print(img)

    X_train = np.expand_dims(np.array(X_train), axis=3)
    uChannel_train = np.expand_dims(np.array(uChannel_train), axis=3)
    vChannel_train = np.expand_dims(np.array(vChannel_train), axis=3)
    X_test = np.expand_dims(np.array(X_test), axis=3)   # y channel (grayscale)
    # print(X_test.shape)
    uChannel_test = np.expand_dims(np.array(uChannel_test), axis=3)
    vChannel_test = np.expand_dims(np.array(vChannel_test), axis=3)


    print(type(X_test))
    print("SHape of X_train- " + str(X_train.shape))
    print("SHape of Y_train- " + str(Y_train.shape))
    print("SHape of X_test- " + str(X_test.shape))
    print("SHape of Y_test- " + str(Y_test.shape))
    # print('<<BREAKPOINT>>')
    # jnsdnj
    # input("Enter")

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    uChannel_test = uChannel_test[holdout:]
    vChannel_test = vChannel_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    # print(X_train.shape[0], X_train.shape[1], type(X_train.shape[2]), X_train.shape[3])
    x = tf.placeholder(tf.float32, shape=(None, int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3])))
    y = tf.placeholder(tf.float32, shape=(None, int(Y_train.shape[1])))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(sess, x, y, X_train, Y_train, X_test, Y_test,
                              nb_epochs, batch_size, learning_rate,
                              rng=rng)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng)
    model_sub, preds_sub = train_sub_out
    print("1")
    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params)
    accuracies['sub'] = acc
    print("?????????????????")
    print(acc)
    print("?????????????????")
    print("11")

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    # os.system('rmdir "D:/sem7/adversarial-attacks/src/cleverhans/cleverhans/__pycache__"')
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    print("Before FastGradientMethod")
    fgsm = FastGradientMethod(model_sub, sess=sess)
    print("After FastGradientMethod")
    print("111")

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    print("TYpe of xTest- "+ str(type(X_test)) + str(X_test.shape))
    x_adv_sub = fgsm.generate(x, **fgsm_par)
    print(type(x_adv_sub), x_adv_sub.shape)
    print("1111")

    
    x_adv_npa = fgsm.generate_np(X_test, **fgsm_par)
    print(type(x_adv_npa), x_adv_npa.shape)
    print(x_adv_npa[1234])
    # print(x_adv_npa.min(), x_adv_npa.max())
    knowLimits(x_adv_npa, 'x_adv_npa')
    print("<< GOTO >>")
    knowLimits(X_test, 'X_test')
    # print(X_test.min(), X_test.max())
    print(type(X_test), X_test.shape)
    print(X_test[1234])
    print("11**11")

    img_x_adv_npa = []
    for i, img in enumerate(X_test):
        yuv_merge = cv2.merge((x_adv_npa[i], uChannel_test[i], vChannel_test[i]))
        # print("Shape of merge- ", yuv_merge.shape)
        img_x_adv_npa.append(cv2.cvtColor(yuv_merge, cv2.COLOR_YUV2RGB))

    img_x_adv_npa = np.array(img_x_adv_npa)

    # print(img_X_test.min(), img_X_test.max())
    # print(img_x_adv_npa.min(), img_x_adv_npa.max())
    knowLimits(img_X_test, 'img_X_test')
    knowLimits(img_x_adv_npa, 'img_x_adv_npa')
    # import pickle

    # with open('x_adv_npa.pickle', 'wb') as f:
    #     pickle.dump(x_adv_npa, f)

    # with sess.as_default():
    # x_adv_npa = fgsm.generate_np(x)
    # print(type(x_adv_npa), x_adv_npa.shape)
    print("DONE !!")

    # fgsm.generate(x, **fgsm_par)

    # x_adv_sub = tf.get_variable(name="x_adv_sub12", initializer = x_adv)# tf.zeros_initializer)
    # x_adv_sub = tf.get_variable(name="x_adv_sub12", shape=[10000, 28, 28, 1], initializer = x_adv)# tf.zeros_initializer)
    # print(x_adv_sub.shape, type(x_adv_sub), x_adv_sub.name)

    # x_adv_sub = x_adv_sub.assign(x_adv)
    # print("NAMED !!! in attacks_tf.py as name= 'x_adv_sub'.")
    # print(x_adv_sub.shape, type(x_adv_sub), x_adv_sub.name)

    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver({"xADV": x_adv_sub})
    # saver = tf.train.Saver([x_adv_sub])
        
    # Save the variables to disk.
    # save_path = saver.save(sess, "srs/model.ckpt")
    # print("shape of x_adv_sub")
    # print(x_adv_sub.shape, type(x_adv_sub))

    print("******* 1 *********")
    
    # print(x_adv_sub)

    # import pickle
    # import dill
    # with open('D:/x'+str(cntdill)+'.pickle', 'wb') as f:
    #     dill.dump(x, f)
    # with open('D:/xAdv'+str(cntdill)+'.pickle', 'wb') as f:
    #     dill.dump(x_adv_sub, f)
    # cntdill+=1

    # print(type(x_adv_sub), x_adv_sub.shape, x_adv_sub)
    # im0 = x_adv_sub[0]
    # Reshape into tf.image.encode_jpeg format
    # image0 = tf.reshape( tf.image.convert_image_dtype(x_adv_sub[0], tf.uint8), [28, 28, 1])

    # # Encode
    # images_encode = tf.image.encode_jpeg(image0)

    # # Create a files name
    # fname = tf.constant('D:/tensorimage.jpeg')

    # # Write files
    # fwrite = tf.write_file(fname, images_encode)
    # ims = (np.reshape(x_adv_sub[0], (28, 28))*255).astype(np.uint8)
    # plt.imshow(ims, interpolation='nearest')
    # plt.show()
    print("******* 2 ***********")

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the subtstitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    # return accuracies#, x, x_adv_sub

    dt = time.strftime('%c')
    fn_X_test = ('X_test_'+dataset+'__'+dt+'.dat').replace(':', '-')
    (img_X_test[holdout:]).dump(fn_X_test)
    fn_x_adv_npa = ('x_adv_npa_'+dataset+'__'+dt+'.dat').replace(':', '-')
    img_x_adv_npa.dump(fn_x_adv_npa)

    return img_x_adv_npa, accuracies


def main(argv=None):
    cntdill =0 
    # acc, x, xAdv = mnist_blackbox(cntdill, nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
    #                learning_rate=FLAGS.learning_rate,
    #                nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
    #                data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
    #                lmbda=FLAGS.lmbda)
    x_adv = mnist_blackbox(cntdill, nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda)

    print("FINAL:- type of x_adv")
    print(type(x_adv))
    # with sess.as_default():
    # x_adv_npa = tf.Session().run(x_adv)
    # print("after eval:-type of x_adv_npa")
    # print(type(x_adv_npa))
    print("\n~Time taken = "+str(time.time()-st)+' secs')
    return x_adv

    # import pickle
    # with open('D:/acc.pickle', 'wb') as f:
    #     pickle.dump(acc, f)


if __name__ == '__main__':
    # General flags
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 32, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 3, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    print('''
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 32, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 3, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')\n''')

st= time.time()
# cntdill=0
print('Start time = '+str(st))
print("Removing cache")
os.system('rmdir /s /q "D:/sem7/adversarial-attacks/src/cleverhans/cleverhans/__pycache__"')
os.system('rmdir /s /q "D:/sem7/adversarial-attacks/src/cleverhans/cleverhans_tutorials/__pycache__"')
os.system('rmdir /s /q "D:/sem7/NIPS_project/src/cleverhans/cleverhans/__pycache__"')
os.system('rmdir /s /q "D:/sem7/NIPS_project/src/cleverhans/cleverhans_tutorials/__pycache__"')
os.system('rmdir /s /q "D:/sem7/NIPS_project/src/cleverhans/cleverhans_tutorials/devtools/__pycache__"')
print("Cache removed")
# sdc
x_adv = tf.app.run()
print("AFTER tf.app.run oO ")
print("FINAL:- type of x_adv")
print(type(x_adv))
x_adv_npa = x_adv.eval()
print("after eval:-type of x_adv_npa")
print(type(x_adv_npa))
print("\nTime taken = "+str(time.time()-st)+' secs\n')