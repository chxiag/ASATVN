"""
This file contains the class which constructs the TensorFlow graph of ASATVN and provides a function for
forecasting.
"""

import numpy as np
import torch
from optimizer import OpenAIAdam

from SATVN import SATVN
from dataHelpers import format_input
from Self_attentive_discriminator import Self_attentive_discriminator

class ASATVN:
    """
    Class for ASATVN.
    """

    def __init__(self, in_seq_length, out_seq_length, d_model, input_dim, h, M, hidden_dim, output_dim, batch_size=1, period=12, n_epochs=100, learning_rate=0.0001, train_x=None, save_file='./asatvn.pt'):
        """
        Constructor
        :param in_seq_length: Sequence length of the inputs.
        :param out_seq_length: Sequence length of the outputs.
        :param d_model:Dimension of the input vector.
        :param input_dim: Dimension of the inputs
        :param h: Number of heads.
        :param M: Number of encoder layers to stack
        :param hidden_dim: Dimension of the hidden units
        :param output_dim: Dimension of the outputs
        :param batch_size: Batch size to use during training. Default: 16
        :param period: seasonal_period
        :param n_epochs: Number of epochs to train over: Default: 100
        :param learning_rate: Learning rate for the Adam algorithm. Default: 0.001
        :param train_x: train_data
        :param save_file: Path and filename to save the model to. Default: './asatvn.pt'
        """

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.d_model = d_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h = h
        self.M = M
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.period = period
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.train_x = train_x
        self.save_file = save_file


        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create the SATVN model
        self.model = SATVN(self.input_dim, self.d_model, self.hidden_dim, self.h, self.M, self.output_dim, self.in_seq_length, self.out_seq_length, self.device)
        # # Use multiple GPUS
        # if torch.cuda.device_count() > 1:
        #     print('Using %d GPUs'%(torch.cuda.device_count()))
        #     self.model = nn.DataParallel(self.model)

        #  Create self-attentive discriminator model
        self.discriminator1 = Self_attentive_discriminator(1, d_model, hidden_dim, 1, d_model, d_model, 1, 1,
                                              attention_size=12, dropout=0, chunk_mode=None, pe='regular',
                                              is_discriminator=True)
        self.discriminator2 = Self_attentive_discriminator(1, d_model, hidden_dim, 1, d_model, d_model, 1, 1,
                                              attention_size=12, dropout=0, chunk_mode=None, pe='regular',
                                              is_discriminator=True)
        self.discriminator1.to(self.device)
        self.discriminator2.to(self.device)
        self.model.to(self.device)


        # Define the optimizer
        lr_warmup = True
        lr_schedule = 'warmup_constant'
        n_updates_total = (self.train_x.__len__() // batch_size) * n_epochs
        self.optimizer_G = OpenAIAdam(self.model.parameters(),
                                 lr=self.learning_rate,
                                 schedule=lr_schedule,
                                 warmup=lr_warmup,
                                 t_total=n_updates_total,
                                 b1=0.9,
                                 b2=0.999,
                                 e=1e-8,
                                 l2=0.01,
                                 vector_l2='store_true',
                                 max_grad_norm=1)
        self.optimizer_D1 = OpenAIAdam(self.discriminator1.parameters(),
                                      lr=self.learning_rate*0.01,
                                      schedule=lr_schedule,
                                      warmup=lr_warmup,
                                      t_total=n_updates_total,
                                      b1=0.9,
                                      b2=0.999,
                                      e=1e-8,
                                      l2=0.01,
                                      vector_l2='store_true',
                                      max_grad_norm=1)
        self.optimizer_D2 = OpenAIAdam(self.discriminator2.parameters(),
                                       lr=self.learning_rate * 0.01,
                                       schedule=lr_schedule,
                                       warmup=lr_warmup,
                                       t_total=n_updates_total,
                                       b1=0.9,
                                       b2=0.999,
                                       e=1e-8,
                                       l2=0.01,
                                       vector_l2='store_true',
                                       max_grad_norm=1)

        self.adversarial_loss = torch.nn.BCELoss()


    def forecast(self, test_x,predict_start):
        """
        Perform a forecast given an input test dataset.
        :param test_x: Input test data in the form [input_seq_length+output_seq_length, batch_size, input_dim]
        :param predict_start: 2*period
        :return: y_hat: The sampled forecast as a numpy array in the form [output_seq_length, batch_size, output_dim]
        """
        self.model.eval()

        # Load model parameters
        checkpoint = torch.load(self.save_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

        with torch.no_grad():

            if type(test_x) is np.ndarray:
                test_x = torch.from_numpy(test_x).type(torch.FloatTensor)  ##（36，3，1）

            # Format the inputs
            test_x = format_input(test_x)##（3，36）
            # Dummy output
            empty_y = torch.empty((self.out_seq_length, test_x[:, :predict_start].shape[1], self.output_dim))  # （12 24 1）
            test_x = test_x.to(self.device)
            empty_y = empty_y.to(self.device)

            # Compute the forecast
            y_hat = self.model(test_x[:, :predict_start], empty_y, is_training=False)
        return y_hat.cpu().numpy()










