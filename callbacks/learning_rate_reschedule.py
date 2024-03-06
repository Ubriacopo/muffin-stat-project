def learning_rate_reschedule(epoch, learning_rate):
    if epoch < 10:
        return learning_rate
    return learning_rate * (1 / (0.9 * epoch))
