import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet

# TODO: Load traffic signs data.
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

# training_file = "/home/carnd/traffic-signs-data/train.p"
# testing_file = "/home/carnd/traffic-signs-data/test.p"
training_file = "./train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
print('X_train.shape={}, y_train.shape={}'.format(X_train.shape, y_train.shape))

# TODO: Split data into training and validation sets.
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
print('after split: X_train.shape={},X_train.shape={}'.format(X_train.shape, y_train.shape))

# TODO: Define placeholders and resize operation.
# placeholder WxH = 32x32, channels =3
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# resize to 227x227 for alexnet input
resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
# one hot encode labels, depth = num classes = 43
nb_classes = 43
one_hot_y = tf.one_hot(y, nb_classes)


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
# 2nd element of shape is the size of output from fc7
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)

# training
loss_operation = tf.reduce_mean(cross_entropy)
rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# accuracy
# Evaluate how well the loss and accuracy of the model for a given dataset.
# tf.argmax returns the index with the largest value across axiss of a tensor
# tf.equal returns the truth value of (x == y) element-wise
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.

# Variables saver
# adds save and restore ops to the graph for all of the variables in the graph
saver = tf.train.Saver()

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/num_examples, total_accuracy / num_examples

# train model

EPOCHS = 2
BATCH_SIZE = 128

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        t0 = time.time()
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_loss, validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    # saves variables to file
    # save_path = saver.save(sess, 'lenet_3channel_dropout')
    # print("Model saved in file: %s" % save_path)
