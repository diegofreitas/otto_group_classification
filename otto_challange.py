__author__ = 'diego'

from sklearn import cross_validation

from lasagne.layers import *
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from helper import *
import pickle



def float32(k):
    return np.cast['float32'](k)

X, y, encoder, scaler = load_train_data('train.csv')
X_test_submission, ids = load_test_data('test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def train(hidden_units, subimission=False):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=561672)
    if subimission:
        X_train = X
        y_train = y

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('gausian0', GaussianNoiseLayer),
               ('dense1', DenseLayer),
               ('dropout0', DropoutLayer),
               ('dense2', DenseLayer),
               ('output', DenseLayer)]
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dense0_num_units=hidden_units[0],
                     dense1_num_units=hidden_units[1],
                     dense2_num_units=hidden_units[2],
                     dropout0_p=0.5,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     update=nesterov_momentum,
                     update_learning_rate=theano.shared(float32(0.03)),
                     update_momentum=theano.shared(float32(0.9)),
                     on_epoch_finished=[
                         AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                         AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     ],
                     eval_size=0.2,
                     verbose=1,
                     max_epochs=15)

    net0.fit(X_train, y_train)

    if subimission:
        make_submission(net0, X_test_submission, ids, encoder, name="nn_%f_%f.csv" % (hidden_units[0], hidden_units[1]))
    else:
        y_prob = net0.predict_proba(X_test)
        print(" -- Finished training.")
        score = logloss_mc(y_test, y_prob)
        print "Score %f" % score
        return score, net0




def main():
    print(" - Start.")
    train([1087, 904], True)
    #evalnn()
    print(" - Finished.")


if __name__ == '__main__':
    main()

