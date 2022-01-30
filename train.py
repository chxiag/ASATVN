"""
A training function for ASATVN.

"""

import numpy as np
import torch
import time
from dataHelpers import format_input

import torch.nn.functional as F

# Set plot_train_progress to True if you want a plot of the forecast after each epoch
plot_train_progress = False
if plot_train_progress:
    import matplotlib.pyplot as plt

def train(ASA, train_x, train_y, validation_x=None, validation_y=None, restore_session=False):
    """
    Train the ASATVN model on a provided dataset.
    In the following variable descriptions, the input_seq_length is the length of the input sequence
    (2*seasonal_period in the paper) and output_seq_length is the number of steps-ahead to forecast
    (seasonal_period in the paper). The n_batches is the total number batches in the dataset. The
    input_dim and output_dim are the dimensions of the input sequence and output sequence respectively
    (in the paper univariate sequences were used where input_dim=output_dim=1).
    :param ASA: A ASATVN object defined by the class in ASATVN.py.
    :param train_x: Input training data in the form [input_seq_length+output_seq_length, n_batches, input_dim]
    :param train_y: Target training data in the form [input_seq_length+output_seq_length, n_batches, output_dim]
    :param validation_x: Optional input validation data in the form [input_seq_length+output_seq_length, n_batches, input_dim]
    :param validation_y: Optional target validation data in the form [input_seq_length+output_seq_length, n_batches, output_dim]
    :param restore_session: If true, restore parameters and keep training, else train from scratch
    :return: training_costs: a list of training costs over the set of epochs
    :return: validation_costs: a list of validation costs over the set of epochs
    """

    # Convert numpy arrays to Torch tensors
    if type(train_x) is np.ndarray:
        train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    if type(train_y) is np.ndarray:
        train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
    if type(validation_x) is np.ndarray:
        validation_x = torch.from_numpy(validation_x).type(torch.FloatTensor)
    if type(validation_y) is np.ndarray:
        validation_y = torch.from_numpy(validation_y).type(torch.FloatTensor)

    # Format inputs
    train_x = format_input(train_x)
    validation_x = format_input(validation_x)

    validation_x = validation_x.to(ASA.device)
    validation_y = validation_y.to(ASA.device)

    # Initialise model with predefined parameters
    if restore_session:
        # Load model parameters
        checkpoint = torch.load(ASA.save_file)
        ASA.model.load_state_dict(checkpoint['model_state_dict'])
        ASA.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Number of samples
    n_samples = train_x.shape[0]

    # List to hold the training costs over each epoch
    training_costs = []
    training_d_costs = []
    validation_costs = []


    # Set in training mode
    ASA.model.train()

    train_window = 3 * ASA.period
    predict_start = 2 * ASA.period

    # Training loop
    for epoch in range(ASA.n_epochs):

        # Start the epoch timer
        t_start = time.time()

        # Print the epoch number
        print('Epoch: %i of %i' % (epoch + 1, ASA.n_epochs))

        # Initial average epoch cost over the sequence
        batch_cost = []
        batch_d_cost = []
        # Counter for permutation loop
        count = 0

        # Permutation to randomly sample from the dataset
        permutation = np.random.permutation(np.arange(0, n_samples-14, ASA.batch_size))

        # Loop over the permuted indexes, extract a sample at that index and run it through the model
        for sample in permutation:
            # Extract a sample at the current permuted index
            input = train_x[sample:sample + ASA.batch_size, :]
            target = train_y[:, sample:sample + ASA.batch_size, :]

            # Send input and output data to the GPU/CPU
            input = input.to(ASA.device)
            target = target.to(ASA.device)

            valid = torch.autograd.Variable(torch.FloatTensor(ASA.batch_size, 1).fill_(1.0), requires_grad=False)  # 16 1
            fake = torch.autograd.Variable(torch.FloatTensor(ASA.batch_size, 1).fill_(0.0), requires_grad=False)  # 16 1

            # Calculate the outputs
            outputs = ASA.model(input=input[:, :predict_start], target=target[predict_start:, :, :],
                                    is_training=True)
            loss = F.mse_loss(input=outputs, target=target[predict_start:, :, :])

            # Train Generator
            # Zero the gradients
            ASA.optimizer_G.zero_grad()

            fake_input = torch.cat((target.squeeze().t()[:, :predict_start], outputs.squeeze().t()),1)  # torch.Size([16, 36])
            valid = valid#.cuda()
            fake = fake#.cuda()

            b2 = 0.1 * ASA.adversarial_loss(ASA.discriminator2(fake_input), valid.squeeze())
            b1 = 0.2 * ASA.adversarial_loss(ASA.discriminator1(outputs.squeeze().t()),valid.squeeze())
            loss = loss + b1 + b2
            loss.backward()
            ASA.optimizer_G.step()
            g_loss = loss.item() / train_window

            # Train the discriminator
            ASA.optimizer_D1.zero_grad()
            ASA.optimizer_D2.zero_grad()

            real_loss1 = ASA.adversarial_loss(ASA.discriminator2(target.squeeze().t(), ),
                                                  valid.squeeze())
            fake_loss2 = ASA.adversarial_loss(ASA.discriminator2(fake_input.detach()),
                                                  fake.squeeze())

            real_loss3 = ASA.adversarial_loss(
                ASA.discriminator1(target[predict_start:, :, :].squeeze().t()), valid.squeeze())
            fake_loss4 = ASA.adversarial_loss(ASA.discriminator1(outputs.squeeze().t().detach()),
                                                  fake.squeeze())

            loss_d = 0.5 * (real_loss1 + fake_loss2) + 0.5 * (real_loss3 + fake_loss4)
            loss_d.backward()
            ASA.optimizer_D1.step()
            ASA.optimizer_D2.step()
            d_loss = loss_d.item()

            batch_cost.append(g_loss)
            batch_d_cost.append(d_loss)
            # Find average cost over sequences and batches
        epoch_cost = np.mean(batch_cost)
        epoch_d_cost = np.mean(batch_d_cost)
        # Calculate the average training cost over the sequence
        training_costs.append(epoch_cost)
        training_d_costs.append(epoch_d_cost)
        # Plot an animation of the training progress
        if plot_train_progress:
            plt.cla()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), target[:, 0, 0])
            temp = outputs.detach()
            plt.plot(np.arange(input.shape[0], input.shape[0] + target.shape[0]), temp[:, 0, 0])
            plt.pause(0.1)

        # Validation tests
        if validation_x is not None:
            ASA.model.eval()
            with torch.no_grad():
                # Calculate the outputs
                y_valid = ASA.model(validation_x[:, :predict_start], validation_y[predict_start:, :, :],
                                        is_training=False)
                # Calculate the loss
                loss = F.mse_loss(input=y_valid, target=validation_y[predict_start:, :, :])  #

                validation_costs.append(loss.item())
            ASA.model.train()

        # Print progress
        print("Average epoch training cost: ", epoch_cost)
        if validation_x is not None:
            print('Average validation cost:     ', validation_costs[-1])
        print("Epoch time:                   %f seconds" % (time.time() - t_start))
        print("Estimated time to complete:   %.2f minutes, (%.2f seconds)" %
              ((ASA.n_epochs - epoch - 1) * (time.time() - t_start) / 60,
               (ASA.n_epochs - epoch - 1) * (time.time() - t_start)))

        # Save a model checkpoint
        best_result = False
        if validation_x is None:
            if training_costs[-1] == min(training_costs):
                best_result = True
        else:
            if validation_costs[-1] == min(validation_costs):
                best_result = True
        if best_result:
            torch.save({
                'model_state_dict': ASA.model.state_dict(),
                'optimizer_state_dict': ASA.optimizer_G.state_dict(),
            }, ASA.save_file)
            print("Model saved in path: %s" % ASA.save_file)
    return training_costs,  training_d_costs, validation_costs
