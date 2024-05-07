from keras.src.callbacks import Callback

# todo move somewhere else this folder will be deleted
class ThresholdStopCallback(Callback):

    def __init__(self, threshold, epoch_start):
        super(ThresholdStopCallback, self).__init__()
        self.threshold = threshold
        self.epoch_start =  epoch_start
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.epoch_start and logs['loss'] > self.threshold:
            self.model.stop_training = True
