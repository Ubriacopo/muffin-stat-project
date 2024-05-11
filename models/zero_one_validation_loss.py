import keras.metrics
import torch
from keras import ops


class ZeroOneLoss(keras.metrics.Metric):
    """
    This class measures the 0-1 loss by counting the miss-classifications.
    Careful that the result displayed during training won't match the real value (keras does the mean on previous batches)
    """

    def __init__(self, name='total_0-1_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.miss_classifications = self.add_weight(name='tp', initializer='zeros', dtype=torch.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.cast(ops.not_equal(y_true, ops.hstack(ops.round(y_pred))), self.dtype)
        # We do not use sample weight
        self.miss_classifications.assign_add(ops.sum(values))

    def result(self):
        return ops.cast(self.miss_classifications, torch.int32)

    def reset_state(self):
        self.miss_classifications.assign(0)


# The following two functions are stateless therefore they do not work as intended on batched data.
# The reported results are valid but for the given batch. Keras does directly the mean on them
# NOTE: In the logs this has to be taken in consideration WITH the Batch Size
def iter_0_1_loss(y_true, y_pred):
    """
    Calculates the zero one loss on the given batch of samples.
    Zero one loss is simply the amount of classifications done wrong of the batch weighted on the whole batch.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: zero one loss
    """
    try:
        if y_true.get_device() != y_pred.get_device():  # I dont know why yet. I will look up
            y_true = y_true.to(device="cuda")
        # The statistical risk with respect to the zero-one loss ℓ(y, yb) = I{yb ̸= y} is therefore defined:
        return ops.hstack(ops.round(y_pred)).not_equal(y_true).sum()

        # Can't use torch.round()

        # y_predicted_max = torch.stack([torch.round(y_pred[i]) for i in range(len(y_pred))])
        # return (ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred)).clone().detach()
        # return torch.tensor(ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred), dtype=torch.float32)
    except Exception as e:
        print("Oops!  That was no valid number.  Try again...")
        print(e)


def zero_one_loss_normalized(y_true, y_pred):
    try:
        if y_true.get_device() != y_pred.get_device():  # I dont know why yet. I will look up
            y_true = y_true.to(device="cuda")
        # The statistical risk with respect to the zero-one loss ℓ(y, yb) = I{yb ̸= y} is therefore defined:
        return ops.mean(ops.hstack(ops.round(y_pred)).not_equal(y_true))

        # Can't use torch.round()

        # y_predicted_max = torch.stack([torch.round(y_pred[i]) for i in range(len(y_pred))])
        # return (ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred)).clone().detach()
        # return torch.tensor(ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred), dtype=torch.float32)
    except Exception as e:
        print("Oops!  That was no valid number.  Try again...")
        print(e)
