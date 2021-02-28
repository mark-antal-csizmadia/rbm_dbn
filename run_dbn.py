from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    """np.random.seed(100)
    indices = np.random.choice(train_imgs.shape[0], 10000, replace=False)
    train_imgs = train_imgs[indices, :]
    train_lbls = train_lbls[indices, :]"""
    print(train_imgs.shape)
    print(train_lbls.shape)

    ''' deep- belief net '''
    
    print ("\nStarting a Deep Belief Net..")
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=100)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")
