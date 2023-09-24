import numpy as np
import matplotlib.pyplot as plt


class Plot_Helper():
    def __init__(self, config):
        self.config = config
        self.loss_train = []
        self.loss_test = []
        self.loss_train_buff = []

        self.figure_file = '../plot/' + self.config.TAG + '/loss_plot.png'

    def record_training(self, loss_train):
        self.loss_train_buff.append(loss_train)
        

    def epoch_summary(self,loss_test, epoch):
        for i in range(self.num_models):
            self.loss_train.append(np.mean(self.loss_train_buff))
            self.loss_test.append(loss_test)

        # reset buffer
        self.loss_train_buff = []
        
        if epoch % self.config.SAVE_PER_EPO == 0:
            fn = '../plot/' + self.config.TAG + '/loss_train.npy'
            np.save(fn, self.loss_train)
            fn = '../plot/' + self.config.TAG + '/loss_test.npy'
            np.save(fn, self.loss_test)

    def plot_loss(self):
        fig = plt.figure(0, figsize=(4, 4))
        plt.clf()
        gs = fig.add_gridspec(1, 1)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(np.array(self.loss_train), 'r', label='Train')
        ax.plot(np.array(self.loss_test), 'b', label='Test')
        plt.title('Loss')


        plt.tight_layout()
        plt.pause(0.001)  # pause a bit so that plots are updated

        fig.savefig(self.figure_file)
