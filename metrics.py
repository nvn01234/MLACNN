from keras.callbacks import Callback
from test import evaluate


class F1score(Callback):

    def __init__(self):
        super(F1score, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        scores = self.model.predict(self.validation_data[0], verbose=False)
        predictions = scores.argmax(-1)
        y_true = self.validation_data[1]
        f1 = evaluate(predictions, y_true)
        print('\t - f1: {:04.2f}'.format(f1))
        logs['f1'] = f1