# Approccio:
 - Faccio ricerca dei parametri migliori in spazio limitato.
 - Prendo i modelli migliori e cerco di sistemarli (3)
 - Trai conclusioni -> se non arriovo a risultati provo a cambiare search space e 
vedere se trovo qualcosa di milgiore ma non vado ad indagare oltre -> Lo search space era troppo esiguo sarebbe la conclusione
 - Se trovo qualcosa che funziona bene e vedo PERCHÈ rispetto agli altri va bene
 - Mi aspetto risultati migliori su CNN e passo a quelle dove ripeto il processo per poi
passare ai modelli pre-trained con architettura omologata

This was looking promising


> Dont use Adam as it performs randomly. The lr parameter is too important
> to apply it during the auto tuining of hyperparameters.
> We had the issue with the loss that could be ~8
> https://arxiv.org/abs/2304.09871 (Leggi paper)
> 
> 


The 550-550 structure behaves like the one with 2k nodes and 1k nodes
Overfitting slowly