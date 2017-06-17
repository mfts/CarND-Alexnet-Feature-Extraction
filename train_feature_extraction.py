import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
traffic_signs = pickle.load('train.p')
print(taffic_signs)

# TODO: Split data into training and validation sets.
X_train, y_train = traffic_signs 
X_valid, y_valid = traffic_signs

nb_classes = len(y_train)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32,32,3))
y = tf.placeholder(tf.float32)


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(x, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
