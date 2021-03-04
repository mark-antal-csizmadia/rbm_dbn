# rbm_dbn

Restricted Boltzmann Machines (RBMs) and Deep Belief Networks (DBNs) from scratch for representation learning on the MNIST dataset.

All of the code has been written based on "A Practical Guide to Training Restricted Boltzmann Machines" by Geoffrey Hinton and "A fast learning algorithm for deep belief nets" by Geoffrey Hinton et al. Both of the papers can be found at literature/.

The documentation of the code generated by Sphinx is located docs/. cd to docs/ then make html, then open index.html at docs/_build/html with your fav browser.

Some of the code is credited to the TAs and lecturers of DD2437 Artificial Neural Networks and Deep Architectures, who delivered an amazing set of lectures and laboratories. Thanks for all, it was really educational! The code was part of a laboratory, the description of which (with much of the theoretical background) is located at literature/.

## Files

1. util.py - Utility file containing activation functions, sampling methods, load/save files, etc.
2. rbm.py - Contains the Restricted Boltzmann Machine class.
3. dbn.py - Contains the Deep Belief Network class.
4. data/train-images-idx3-ubyte - MNIST training images
5. data/train-labels-idx1-ubyte - MNIST training labels
6. data/t10k-images-idx3-ubyte - MNIST test images
data/t10k-labels-idx1-ubyte - MNIST test labels
7. trained_rbm/ - Directory to store trained RBM model
8. single_rbm/ - Directory to store figures of the reconstruction losses of different RBMs.
9. rbm_viz/ - Directory to store the learned weights of different RBMs.
10. rbm_dbn.ipynb - Notebook for a walkthrough and demo.
11. litrature/ - Papers and documents that the code is based on.
12. dbn_mp4/ - Directory to store an animation of generating digits from the trained DBN.
13. docs/ - Sphinx docs of code.
    
## TODO

Implement the wake-sleep algorithm to fine-tune all the parameters of the DBN. 
1. dbn.train_wakesleep_finetune() - main method for wake-sleep learning 
2. rbm.update_generate_params() - updates the generative parameters (directed)
3. rbm.update_recognize_params() - updates the recognition parameters (directed)	

Implement momentum update for more efficient gradient-based optimization
rbm.update_params(v_0, h_0, v_k, h_k)

Improve weight initialization in rbm and dbn - right now just random normal with random seed set.

## Time to run

A rough estimate of how much running time can be expected for each section in the notebook.

Training a single RBM will be in the order 10-20 minutes for the whole training set.
Training a DBN, that involves training three seperate RBMs, it is in the order of three times longer than training a single RBM, so around 30 to 90 minutes.
The wake-sleep fine-tuning will (when completed) take around 30 to 60 minutes.


