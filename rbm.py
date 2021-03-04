from util import *
import numpy as np 
import matplotlib.pyplot as plt


class RestrictedBoltzmannMachine():
    """
    Restricted Boltzmann Machine class.
    For more details : A Practical Guide to Training Restricted Boltzmann Machines
    Available at: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

    Attributes
    ----------
    ndim_visible : int
        Number of units in visible layer.
    ndim_hidden : int
        Number of units in hidden layer.
    is_bottom : bool
        True only if this rbm is at the bottom of the stack in a deep belief net.
        Used to interpret visible layer as image data with dimensions "image_size".
    image_size : list
        Image dimension for visible layer with elements for width and height.
    is_top : bool
        True only if this rbm is at the top of stack in deep beleif net.
        Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
    n_labels : int
        Number of label categories. It is 10 in this project for the 10 hand-written digits in MNIST
    batch_size : int
        Size of mini-batch.
    delta_bias_v : numpy.ndarray
        The gradient of the bias parameter of the visible layer. Used in the parameter update rule.
        Of shape (size of visible layer, ).
    delta_weight_vh : numpy.ndarray
        The gradient of the learnable weight parameters between the visible and the hidden layer.
        Of shape (size of visible layer, size of hidden layer).
    delta_bias_h : numpy.ndarray
        The gradient of the bias parameter of the hidden layer. Used in the parameter update rule.
        Of shape (size of hidden layer, ).
    bias_v : numpy.ndarray
        The bias parameter of the visible layer. Used in the parameter update rule.
        Of shape (size of visible layer, ).
    weight_vh : numpy.ndarray
        The learnable weight parameters between the visible and the hidden layer.
        Of shape (size of visible layer, size of hidden layer).
    bias_h : numpy.ndarray
        The bias parameter of the hidden layer. Used in the parameter update rule.
        Of shape (size of hidden layer, ).
    delta_weight_v_to_h : numpy.ndarray
        The gradient of the directed learnable weight parameters between the visible and the hidden layer
        when the RBM is a Bayesian network, that is, when it is not the top layer of a DBN.
        Of shape (size of visible layer, size of hidden layer).
    delta_weight_h_to_v : numpy.ndarray
        The gradient of the directed learnable weight parameters between the hidden and the visible layer
        when the RBM is a Bayesian network, that is, when it is not the top layer of a DBN.
        Of shape (size of hidden layer, size of visible layer).
    weight_v_to_h : numpy.ndarray
        The directed learnable weight parameters between the visible and the hidden layer
        when the RBM is a Bayesian network, that is, when it is not the top layer of a DBN.
        Of shape (size of visible layer, size of hidden layer).
    weight_h_to_v : numpy.ndarray
        The directed learnable weight parameters between the hidden and the visible layer
        when the RBM is a Bayesian network, that is, when it is not the top layer of a DBN.
        Of shape (size of hidden layer, size of visible layer).
    learning_rate : float
        The learning rate in the parameter update rule, set to 0.01.
    momentum : float
        The momentum update rule parameter that defines the running average weighting, set to 0.7.
    print_period : int
        Print out at this rate while training.

    Methods
    -------
    cd1(self, visible_trainset, n_iterations)
        Contrastive Divergence with k=1 full alternating Gibbs sampling.
    update_params(self, v_0, h_0, v_k, h_k)
        Parameter update. TODO: momentum update
    get_h_given_v(self, visible_minibatch)
        Compute probabilities p(h|v) and activations h ~ p(h|v).
        Uses undirected weight "weight_vh" and bias "bias_h"
    get_v_given_h(self, hidden_minibatch)
        Compute probabilities p(v|h) and activations v ~ p(v|h)
        Uses undirected weight "weight_vh" and bias "bias_v"
    untwine_weights(self)
        Untying weights of RBM, used when part of DBN.
    get_h_given_v_dir(self, visible_minibatch)
        Compute probabilities p(h|v) and activations h ~ p(h|v). Used when RBM is in directed Bayesian network
        of DBN. Uses directed weight "weight_v_to_h" and bias "bias_h"
    get_v_given_h_dir(self, hidden_minibatch)
        Compute probabilities p(v|h) and activations v ~ p(v|h). Used when RBM is in directed Bayesian network
        of DBN. Uses directed weight "weight_h_to_v" and bias "bias_v"
    update_generate_params(self, inps, trgs, preds)
        TODO: Update generative weight "weight_h_to_v" and bias "bias_v"
    update_recognize_params(self, inps, trgs, preds)
        TODO: Update recognition weight "weight_v_to_h" and bias "bias_h"
    """
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10,
                 batch_size=10):
        """ Initialize the RBM.

        Parameters
        ----------
        ndim_visible : int
            Number of units in visible layer.
        ndim_hidden : int
            Number of units in hidden layer.
        is_bottom : bool
            True only if this rbm is at the bottom of the stack in a deep belief net.
            Used to interpret visible layer as image data with dimensions "image_size".
        image_size : list
            Image dimension for visible layer with elements for width and height.
        is_top : bool
            True only if this rbm is at the top of stack in deep beleif net.
            Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
        n_labels : int
            Number of label categories. It is 10 in this project for the 10 hand-written digits in MNIST
        batch_size : int
            Size of mini-batch.

        Returns
        -------
        None
        """
        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.is_bottom = is_bottom
        if is_bottom : self.image_size = image_size
        self.is_top = is_top
        if is_top : self.n_labels = 10
        self.batch_size = batch_size           
        self.delta_bias_v = 0
        self.delta_weight_vh = 0
        self.delta_bias_h = 0
        np.random.seed(111)
        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        np.random.seed(222)
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))
        np.random.seed(333)
        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0        
        self.weight_v_to_h = None
        self.weight_h_to_v = None
        self.learning_rate = 0.01
        self.momentum = 0.7
        self.print_period = 1

        # receptive-fields. Only applicable when visible layer is input data
        np.random.seed(444)
        self.rf = {
            # iteration period to visualize
            "period": 5,
            # size of the grid
            "grid": [5, 5],
            # pick some random hidden units
            "ids": np.random.randint(0, self.ndim_hidden, 25)
            }
        
        return

    def cd1(self, visible_trainset, n_iterations):
        """ Contrastive Divergence with k=1 full alternating Gibbs sampling

        Parameters
        ----------
        visible_trainset : numpy.ndarray
            Training data for this rbm, shape is (size of training set, size of visible layer)
        n_iterations : int
            The number of iterations of learning, technically it is the epoch number.

        Returns
        -------
        loss_history : list
            The mean reconstruction loss per epoch.
        """
        print("learning CD1")
        n_samples = visible_trainset.shape[0]
        rounds = int(n_samples/self.batch_size)
        loss = []
        loss_history = []

        for it in range(n_iterations):
            for idx in range(rounds):
                start_index = int(idx % rounds)
                end_index = int((start_index+1)*self.batch_size)
                v_0 = visible_trainset[start_index*self.batch_size:end_index, :]
                # Run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
                _, h_0 = self.get_h_given_v(v_0)
                _, v_k = self.get_v_given_h(h_0)
                _, h_k = self.get_h_given_v(v_k)
                # Update the parameters using function 'update_params'
                self.update_params(v_0, h_0, v_k, h_k)
                _, h_0 = self.get_h_given_v(v_0)
                _, v_k = self.get_v_given_h(h_0)
                loss.append(np.linalg.norm(v_0 - v_k))

            loss_history.append(np.mean(loss))
            loss = []

            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it,
                       ndim_hidden=self.ndim_hidden,
                       grid=self.rf["grid"])

            if it % self.print_period == 0:
                print("iteration=%7d Reconstruction loss=%4.4f" % (it+1, loss_history[it]))

        return loss_history

    def update_params(self, v_0, h_0, v_k, h_k):
        """ Parameter update. TODO: momentum update

        Parameters
        ----------
        v_0 : numpy.ndarray
            Activities or probabilities of visible layer (data to the rbm).
            Of shape: (size of mini-batch, size of respective layer)
        h_0 : numpy.ndarray
            Activities or probabilities of hidden layer.
            Of shape: (size of mini-batch, size of respective layer)
        v_k : numpy.ndarray
            Activities or probabilities of visible layer.
            Of shape: (size of mini-batch, size of respective layer)
        h_k : numpy.ndarray
            Activities or probabilities of hidden layer.
            Of shape: (size of mini-batch, size of respective layer)

        Returns
        -------
        None
        """
        # Get the gradients from the arguments and update the weight and bias parameters
        self.delta_bias_v = self.learning_rate * (np.sum(v_0-v_k, axis=0))
        self.delta_weight_vh = self.learning_rate * (np.dot(np.transpose(v_0), h_0) - np.dot(np.transpose(v_k), h_k))
        self.delta_bias_h = self.learning_rate * (np.sum(h_0-h_k, axis=0))

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

    def get_h_given_v(self, visible_minibatch):
        """ Compute probabilities p(h|v) and activations h ~ p(h|v).
        Uses undirected weight "weight_vh" and bias "bias_h"

        Parameters
        ----------
        visible_minibatch : numpy.ndarray
            The data on the visible layer, of shape is (size of mini-batch, size of visible layer)

        Returns
        -------
        h_given_v, h : tuple
            Tuple of p(h|v) (probability distribution of the hidden pattern given the visible pattern),
            h (the hidden pattern after sampling from p(h|v)),
            both are shaped (size of mini-batch, size of hidden layer)
        """
        # For untying weights in DBNs.
        assert self.weight_vh is not None
        n_samples = visible_minibatch.shape[0]
        # Compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)
        h_given_v = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        h = sample_binary(h_given_v)

        return h_given_v, h

    def get_v_given_h(self, hidden_minibatch):
        """ Compute probabilities p(v|h) and activations v ~ p(v|h)
        Uses undirected weight "weight_vh" and bias "bias_v"

        Parameters
        ----------
        hidden_minibatch : numpy.ndarray
            The data on the hidden layer, of shape is (size of mini-batch, size of hidden layer)

        Returns
        -------
        v_given_h, v : tuple
            Tuple of p(v|h) (probability distribution of the visible pattern given the hidden pattern),
            v (the visible pattern after sampling from p(v|h)),
            both are shaped (size of mini-batch, size of visible layer)
        """
        # For untying weights in DBNs.
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        # Here visible layer has both data and labels. Compute total input for each unit (identical for both cases),
        # and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
        # Then, for both parts, use the appropriate activation function to get probabilities and a sampling method
        # to get activities. The probabilities as well as activities can then be concatenated back into a normal
        # visible layer.
        if self.is_top:
            # Compute probabilities and activations (samples from probabilities) of visible layer.
            # Stand-alone RBMs do not contain labels in visible layer.
            support = self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T)

            # Compute probabilities only for visible layer
            v_given_h = np.zeros(shape=support.shape)
            v = np.ndarray(shape=support.shape)

            v_given_h[:, :-self.n_labels] = sigmoid(support[:, :-self.n_labels])
            v[:, :-self.n_labels] = sample_binary(v_given_h[:, :-self.n_labels])
            v_given_h[:, -self.n_labels:] = softmax(support[:, -self.n_labels:])
            v[:, -self.n_labels:] = sample_categorical(v_given_h[:, -self.n_labels:])
        else:
            # Compute probabilities and activations (samples from probabilities) of visible layer.
            v_given_h = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T))
            v = sample_binary(v_given_h)

        return v_given_h, v

    def untwine_weights(self):
        """ Untying weights of RBM, used when part of DBN.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """ Compute probabilities p(h|v) and activations h ~ p(h|v). Used when RBM is in directed Bayesian network
        of DBN. Uses directed weight "weight_v_to_h" and bias "bias_h"

        Parameters
        ----------
        visible_minibatch : numpy.ndarray
            The data on the visible layer, of shape is (size of mini-batch, size of visible layer)

        Returns
        -------
        h_given_v, h : tuple
            Tuple of p(h|v) (probability distribution of the hidden pattern given the visible pattern),
            h (the hidden pattern after sampling from p(h|v)),
            both are shaped (size of mini-batch, size of hidden layer)
        """
        # For untying weights in DBNs.
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]
        # Perform same computation as the function 'get_h_given_v' but with directed connections.
        h_given_v_dir = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h))
        h = sample_binary(h_given_v_dir)

        return h_given_v_dir, h

    def get_v_given_h_dir(self, hidden_minibatch):
        """ Compute probabilities p(v|h) and activations v ~ p(v|h). Used when RBM is in directed Bayesian network
        of DBN. Uses directed weight "weight_h_to_v" and bias "bias_v"

        Parameters
        ----------
        hidden_minibatch : numpy.ndarray
            The data on the hidden layer, of shape is (size of mini-batch, size of hidden layer)

        Returns
        -------
        v_given_h, v : tuple
            Tuple of p(v|h) (probability distribution of the visible pattern given the hidden pattern),
            v (the visible pattern after sampling from p(v|h)),
            both are shaped (size of mini-batch, size of visible layer)
        """
        # For untying weights in DBNs.
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]

        # Here visible layer has both data and labels. Compute total input for each unit (identical for both cases),
        # and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
        # Then, for both parts, use the appropriate activation function to get probabilities and a sampling method
        # to get activities.
        # The probabilities as well as activities can then be concatenated back into a normal visible layer.
        if self.is_top:
            # Note that even though this function performs same computation as 'get_v_given_h' but with directed
            # connections, this case should never be executed: when the RBM is a part of a DBN and is at the top,
            # it will have not have directed connections.
            # Appropriate code here is to raise an error
            raise Exception("ERROR: No directed connections when RBM is a part of DBN and is at the top")
        else:
            # Performs same computation as the function 'get_v_given_h' but with directed connections.
            v_given_h_dir = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_h_to_v))
            activated_samples = sample_binary(v_given_h_dir)
            
        return v_given_h_dir, activated_samples     
        
    def update_generate_params(self, inps, trgs, preds):
        """ TODO: Update generative weight "weight_h_to_v" and bias "bias_v"

        Parameters
        ----------
        inps : numpy.ndarray
            Activities or probabilities of input unit. Of shape: (size of mini-batch, size of respective layer).
        trgs : numpy.ndarray
            Activities or probabilities of output unit (target).
            Of shape: (size of mini-batch, size of respective layer).
        preds : numpy.ndarray
            Activities or probabilities of output unit (prediction).
            Of shape: (size of mini-batch, size of respective layer).

        Returns
        -------
        None
        """
        # TODO: Find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self, inps, trgs, preds):
        """ TODO: Update recognition weight "weight_v_to_h" and bias "bias_h"

        Parameters
        ----------
        inps : numpy.ndarray
            Activities or probabilities of input unit. Of shape: (size of mini-batch, size of respective layer).
        trgs : numpy.ndarray
            Activities or probabilities of output unit (target).
            Of shape: (size of mini-batch, size of respective layer).
        preds : numpy.ndarray
            Activities or probabilities of output unit (prediction).
            Of shape: (size of mini-batch, size of respective layer).

        Returns
        -------
        None
        """
        # TODO: Find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
