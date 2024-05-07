- Non fare distinzioni su augmented e normal come folder. Alla fine sono sempre 2layer DNN. Il pre processing
  è semplicemente un bonus diciamo. Che volendo possiamo rendere indipendente dall'architettura.

- Usare i logit
- Fare che la loss data non siqa solo stringa ma possiblmente
anche una loss function cosi posso settare from logit etc
- Cambia nome a ChannelsLastFixAugmentedNaiveDNNModelFamily
- zero-one loss is on CPU avoid it.
- Sistema il file di conv_net_family. Ne farei diversi sinceramente