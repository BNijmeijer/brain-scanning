# brain-scanning
Code used for creating and testing the CNNs and RNNs used in our paper.

## Data acquisition
When running main.py for the first time, the code will generate the preprocessed data as used in all networks, and save these as .npz files in your local repo folder.
Any subsequent runs of main.py will utilize these .npz files, in order to speed up data preprocessing.

By default, the code will run an Intra-Cross comparison on a tuned RNN. Additional functions exist for cross-validation of the RNN, and plotting the training accuracies of the Intra data on a tuned RNN.

## File structure
The code is split up among three files:

- `main.py`, and model training functions;
- `models.py`, containing all model definitions;
- `preprocessing.py`, containing all code regarding data loading preprocessing.