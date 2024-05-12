 Per k-fold cv:
 
- Prendo train e test e li mischio insieme.
- Faccio 5-fold-cv senza early stopping?
  - Uso un fold per fare validazione cosi da poter fare early stopping?
    - Non uso early stopping e bon si risolve.

https://scikit-learn.org/stable/modules/cross_validation.html


# kfold

In K-fold posso usare early stopping in questo modo:
- Il primo modello esegue early stopping come deciso (cold type of parameter search)
- I modelli successivi terminano all'iterazione del primo
- https://stats.stackexchange.com/questions/298084/k-fold-cross-validation-for-choosing-number-of-epochs

In alternativa scegli un set di epcohs [20, 30, 40]


I think in both XGBoost and LightGBM, the CV will use the average scores from all folds, and use this for the early stopping. Therefore, best_iteration is the same in all folds. I think this is more stable since the average_score is computed over all data samples, not just over the current fold.
https://datascience.stackexchange.com/questions/74351/what-is-the-proper-way-to-use-early-stopping-with-cross-validation


> Validation data sets can be used for regularization by early stopping (stopping training when the error on the validation data set increases, as this is a sign of over-fitting to the training data set).[6] 
>  ~ Neural Networks: Tricks of the Trade