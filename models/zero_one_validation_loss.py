import torch
from keras import ops


def zero_one_loss(y_true, y_pred):
    """
    Calculates the zero one loss on the given batch of samples.
    Zero one loss is simply the amount of classifications done wrong of the batch weighted on the whole batch.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: zero one loss
    """
    try:
        y_predicted_max = torch.stack([torch.argmax(y_pred[i]) for i in range(len(y_pred))])
        return (ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred)).clone().detach()
        # return torch.tensor(ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred), dtype=torch.float32)
    except Exception as e:
        print("Oops!  That was no valid number.  Try again...")
        print(e)


def zero_one_loss_binary(y_true, y_pred):
    """
    Calculates the zero one loss on the given batch of samples.
    Zero one loss is simply the amount of classifications done wrong of the batch weighted on the whole batch.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: zero one loss
    """
    try:
        y_predicted_max = torch.stack([torch.round(y_pred[i]) for i in range(len(y_pred))])
        return (ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred)).clone().detach()
        # return torch.tensor(ops.not_equal(y_predicted_max, y_true).sum() / len(y_pred), dtype=torch.float32)
    except Exception as e:
        print("Oops!  That was no valid number.  Try again...")
        print(e)
