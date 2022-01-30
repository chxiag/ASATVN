"""
Code to Evaluate of using the Mean Absolute Scaled Error (MASE) and the Symmetric Mean Absolute Percentage Error (SMAPE) for a given test set.

"""

import numpy as np
import torch
from dataHelpers import format_input
from calculateError import calculate_error

def evaluate(ASA, test_x, test_y, return_lists=False):
    """
    Calculate various error metrics on a test dataset
    :param ASA: A ASATVN object defined by the class in ASATVN.py
    :param test_x: Input test data in the form [in_seq_length+out_seq_length, n_batches, input_dim]
    :param test_y: target data in the form [in_seq_length+out_seq_length, n_batches, input_dim]
    :return: mase: Mean absolute scaled error
    :return: smape: Symmetric absolute percentage error
    :return: nrmse: Normalised root mean squared error
    """
    ASA.model.eval()
    predict_start = 24

    # Load model parameters
    checkpoint = torch.load(ASA.save_file, map_location=ASA.device)
    ASA.model.load_state_dict(checkpoint['model_state_dict'])
    ASA.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])

    with torch.no_grad():
        if type(test_x) is np.ndarray:
            test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        if type(test_y) is np.ndarray:
            test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

        # Format the inputs
        test_x = format_input(test_x)

        # Send to CPU/GPU
        test_x = test_x.to(ASA.device)
        test_y = test_y.to(ASA.device)

        # Number of batch samples
        n_samples = test_x.shape[0]

        # Inference
        y_pred_list = []
        # Compute outputs for a mixture density network output
        # Compute outputs for a linear output
        y_pred = ASA.model(test_x[:, :predict_start], test_y[predict_start:, :, :], is_training=False)

        mase_list = []
        smape_list = []
        nrmse_list = []
        for i in range(n_samples):
            mase, se, smape, nrmse = calculate_error(y_pred[:, i, :].cpu().numpy(), test_y[predict_start:, :, :][:, i,
                                                                                    :].cpu().numpy())  # y_pred 和test_y 最后的shape是什么样子的
            mase_list.append(mase)
            smape_list.append(smape)
            nrmse_list.append(nrmse)
        # writer.close()
        mase = np.mean(mase_list)
        smape = np.mean(smape_list)
        nrmse = np.mean(nrmse_list)

    if return_lists:
        return np.ndarray.flatten(np.array(mase_list)), np.ndarray.flatten(np.array(smape_list)), np.ndarray.flatten(
            np.array(nrmse_list))
    else:
        return mase, smape, nrmse
