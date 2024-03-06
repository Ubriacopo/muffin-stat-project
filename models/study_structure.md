# How a "Study" is structured
While not a study in the academic sense of the term I simply note the process
that followed the creation of a NN structure via theoretical considerations, 
best practices and various tweaking after training has been done. 

> It is always best to start with something simple and enrich it rather than trying
> a complex solution directly. (As seen in class)

## Steps
The process I apply to create a structure and check if it could be valid is the following one:
- <b>[def architecture]</b>: The structure of the Network
- <b>[eval architecture]</b>: Via training and test we see performance
- Save the history (process) on the data and if necessary tweak by going back to step <b>[def architecture]</b>
- Search confirmations of the model via K-fold CV (so we can be sure we weren't lucky)

### Why do we do K-fold CV only at last?
Reason is simply computational times.

## Study notebook structure
