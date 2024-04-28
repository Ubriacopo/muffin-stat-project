# out of date.

# Project Structure
We approach the project and structure it following the *common reuse principle* as I believe it to be the best practice to opt for. It can be argued that the project is indeed structurally simple enough to divide it in the classic MVC structure. But navigation as I present it should feel simpler to follow in CRP.

- **/%model_type%**
  - **/%model_type%.py**: Contains the definition the model class and auxiliary functions relative to it. These definitions are used outside of this folder (it can be seen as some sort of public package).
  - **/%model_type%.ipynb**: Jupyter notebook where the process of thought to reach the model was practiced. It contains some considerations but is confined entirely on the model defined in the folder.
  - **/mod-gen/**: Contains the generated model values (matrices for the NN)
- **/data/**: Contains the functions to retrieve the data from Kaggle.