```json

Value             |Best Value So Far |Hyperparameter
SGD               |SGD               |optimizer
3                 |1                 |layers
256               |256               |units_0
False             |False             |dropout_0
512               |256               |units_1
False             |True              |dropout_1
32                |None              |units_2
True              |None              |dropout_2
```


This was looking promising


> Dont use Adam as it performs randomly. The lr parameter is too important
> to apply it during the auto tuining of hyperparameters.
> We had the issue with the loss that could be ~8