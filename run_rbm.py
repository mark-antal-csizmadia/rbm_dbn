from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    indices = np.random.choice(train_imgs.shape[0], 1000, replace=False)
    train_imgs = train_imgs[indices, :]
    train_lbls = train_lbls[indices, :]
    print(train_imgs.shape)
    print(train_lbls.shape)


    ''' restricted boltzmann machine '''
    print ("\nStarting a Restricted Boltzmann Machine..")
    # Change this list depending on the task
    hidden_dims = [200, 300, 400, 500]
    epochs = 30
    loss_history = np.zeros((len(hidden_dims), epochs))
    i = 0
    for hidden in hidden_dims:
        print(hidden)
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=hidden,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=15
        )
    
        loss_history[i] = rbm.cd1(visible_trainset=train_imgs, n_iterations=epochs)
        i += 1
    for plots in range(len(hidden_dims)):
        plt.plot(range(epochs), loss_history[plots], label = hidden_dims[plots])
    plt.title("Reconstruction loss over epochs")
    plt.legend(title = "ndim_hidden")
    plt.show()