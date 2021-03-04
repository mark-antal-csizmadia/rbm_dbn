from util import *
from rbm import RestrictedBoltzmannMachine


class DeepBeliefNet():
    """
    Deep Belief Network class.
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets.
    Available at: https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    The DBN network architecture from the paper looks as follows
    [top] <---> [pen] + [lbl] ---> [hid] ---> [vis]
    where
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible

    Attributes
    ----------
    rbm_stack : dict
        A dictionary for defining the network architecture, see in paper.
    sizes : dict
        Dictionary of layer names and dimensions
    image_size : list
        Image dimension of data, has 2 elements for width, height
    batch_size : int
        Size of mini-batch for accumulating gradients during parameter update
    n_gibbs_recog : int
        Number of iterations in Gibbs sampling during CD when in recognition mode
    n_gibbs_gener : int
        Number of iterations in Gibbs sampling during CD when in geenrative mode
    n_gibbs_wakesleep : int
        Number of iterations in Gibbs sampling during CD when in wake-sleep mode (for the supervised fine-tuning params)
    print_period : int
        Print out at this rate during training

    Methods
    -------
    recognize(self, true_img, true_lbl)
        Recognize/Classify the data into label categories and calculate the accuracy
    generate(self, true_lbl, name)
        Generate data from labels
    train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations)
        Greedy layer-wise training by stacking RBMs.
        This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.
    train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations)
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.
    loadfromfile_rbm(self, loc, name)
        Loads saved RBM parameters.
    savetofile_rbm(self, loc, name)
        Saves RBM parameters.
    loadfromfile_dbn(self, loc, name)
        Loads saved DBN parameters.
    savetofile_dbn(self, loc, name)
        Saves DBN parameters.
    """
    def __init__(self, sizes, image_size, n_labels, batch_size):
        """ Initialize DBN.

        Parameters
        ----------
        sizes : dict
            Dictionary of layer names and dimensions
        image_size : list
            Image dimension of data, has 2 elements for width, height
        n_labels : int
            Number of labels (i.e.: number of classes) in data.
        batch_size : int
            Size of mini-batch for accumulating gradients during parameter update

        Returns
        -------
        None
        """
        self.rbm_stack = {
            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"],
                                                   ndim_hidden=sizes["hid"],
                                                   is_bottom=True,
                                                   image_size=image_size,
                                                   batch_size=batch_size),
            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"],
                                                   ndim_hidden=sizes["pen"],
                                                   batch_size=batch_size),
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"],
                                                        ndim_hidden=sizes["top"],
                                                        is_top=True,
                                                        n_labels=n_labels,
                                                        batch_size=batch_size)
        }
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5
        self.print_period = 2000
        
        return

    def recognize(self, true_img, true_lbl):
        """ Recognize/Classify the data into label categories and calculate the accuracy

        Parameters
        ----------
        true_img : numpy.ndarray
            visible data shaped (number of samples, size of visible layer)
        true_lbl : numpy.ndarray
            true labels shaped (number of samples, size of label layer).
            Used only for calculating accuracy, not driving the net

        Returns
        -------
        None
        """
        n_samples = true_img.shape[0]
        n_lables = true_lbl.shape[1]
        vis = true_img # visible layer gets the image data
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # Fix the image data in the visible layer and drive the network bottom to top.
        # In the top RBM, run alternating Gibbs sampling \
        # and read out the labels - 'predicted_lbl' is the predicted labels.
        # Inferring entire train/test set may require too much compute memory (depends on your system).
        # In that case, divide into mini-batches.
        _, hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
        _, pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid)

        pen_lbl = np.concatenate((pen, lbl), axis=1)

        for _ in range(self.n_gibbs_recog):
            _, top = self.rbm_stack['pen+lbl--top'].get_h_given_v(pen_lbl)
            _, pen_lbl = self.rbm_stack['pen+lbl--top'].get_v_given_h(top)

        predicted_lbl = pen_lbl[:, -n_lables:]

        print("accuracy = %.2f%%" % (100.*np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))))
        
        return

    def generate(self, true_lbl, name):
        """ Generate data from labels

        Parameters
        ----------
        true_lbl : numpy.ndarray
            true labels shaped (number of samples, size of label layer).
        name : str
            used for saving a video of generated visible activations

        Returns
        -------
        None
        """
        n_sample = true_lbl.shape[0]
        n_labels = true_lbl.shape[1]

        records = []        
        fig, ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl
        np.random.seed(555)
        dummy_img = np.random.choice([0, 1], self.sizes['vis']).reshape(-1, self.sizes['vis'])

        # Fix the label in the label layer and run alternating Gibbs sampling in the top RBM.
        # From the top RBM, drive the network top to the bottom visible layer.
        _, hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(dummy_img)
        _, pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid)

        pen_lbl = np.concatenate((pen, lbl), axis=1)

        for _ in range(self.n_gibbs_gener):
            _, top = self.rbm_stack['pen+lbl--top'].get_h_given_v(pen_lbl)
            _, pen_lbl = self.rbm_stack['pen+lbl--top'].get_v_given_h(top)
            pen_lbl[:, -n_labels:] = lbl[:, :] #?

            pen = pen_lbl[:, :-n_labels]
            _, hid = self.rbm_stack['hid--pen'].get_v_given_h_dir(pen)
            _, vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(hid)
            
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1,
                                        animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig, records).save("dbn_mp4/%s.generate%d.mp4" % (name, np.argmax(true_lbl)))

        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        """ Greedy layer-wise training by stacking RBMs. 
        This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Parameters
        ----------
        vis_trainset : numpy.ndarray
            visible data shaped (size of training set, size of visible layer)
        lbl_trainset : numpy.ndarray
            label data shaped (size of training set, size of label layer)
        n_iterations : int
            number of iterations of learning (each iteration learns a mini-batch)
        
        Returns
        -------
        None
        """
        try:
            self.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()
            self.loadfromfile_rbm(loc="trained_rbm", name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            self.loadfromfile_rbm(loc="trained_rbm", name="pen+lbl--top")

        except IOError :
            # Use CD-1 to train all RBMs greedily
            print("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """            
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")
            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """            
            self.rbm_stack["vis--hid"].untwine_weights()   
            _, hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
            self.rbm_stack["hid--pen"].cd1(hid, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")
            print("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            _, pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid)
            pen_lbl = np.concatenate((pen, lbl_trainset), axis=1)
            self.rbm_stack["pen+lbl--top"].cd1(pen_lbl, n_iterations)

            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):
        """ Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Parameters
        ----------
        vis_trainset : numpy.ndarray
            visible data shaped (size of training set, size of visible layer)
        lbl_trainset : numpy.ndarray
            label data shaped (size of training set, size of label layer)
        n_iterations : int
            number of iterations of learning (each iteration learns a mini-batch)

        Returns
        -------
        None
        """
        
        print("\ntraining wake-sleep..")

        try:
            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):
                # [TODO] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps,
                #  also store neccessary information for learning this RBM.

                # [TODO] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO] compute predictions : compute generative predictions from wake-phase activations,
                #  and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations,
                # we use them only to learn the directed connections.
                
                # [TODO] update generative parameters : here you will only use 'update_generate_params'
                #  method from rbm class.

                # [TODO] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO] update generative parameters : here you will only use 'update_recognize_params' method
                #  from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    def loadfromfile_rbm(self, loc, name):
        """ Loads saved RBM parameters.

        Parameters
        ----------
        loc : str
            Path to saved model params.
        name : str
            Name of saved model

        Returns
        -------
        None
        """
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return
        
    def savetofile_rbm(self, loc, name):
        """ Saves RBM parameters.

        Parameters
        ----------
        loc : str
            Path to saved model params.
        name : str
            Name of saved model

        Returns
        -------
        None
        """
        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc, name), self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self, loc, name):
        """ Loads saved DBN parameters.

        Parameters
        ----------
        loc : str
            Path to saved model params.
        name : str
            Name of saved model

        Returns
        -------
        None
        """
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name))
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name))
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name))
        print("loaded rbm[%s] from %s" % (name, loc))
        return
        
    def savetofile_dbn(self, loc, name):
        """ Saves DBN parameters.

        Parameters
        ----------
        loc : str
            Path to saved model params.
        name : str
            Name of saved model

        Returns
        -------
        None
        """
        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return
