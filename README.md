# ASATVN

PyTorch implementation of ASATVN described in the paper entitled 
"Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting" 
by Changxia Gao, Ning Zhang, Youru Li and Huaiyu Wan.


The key benifits of ASATVN are:
1. It is a time-variant model, as opposed to a time-invariant model (In the paper we show that RNN and CNN models are time-invariant).
2. It's interleaved outputs assist with convergence and mitigating vanishing-gradient problems.
3. The newly proposed Truncated Cauchy self-attention block makes the time-variant neural network more sensitive to
the local context of time series.
4. Two self-attentive discriminators are attached to time-variant neural network to offer more realistic and continuous long-term forecast
results, respectively.
5. It is shown to out-perform state of the art deep learning models and statistical models.

## Files


- ASATVN.py: Contains the main class for ASATVN (Adversarial Self-Attentive Time-Variant Network).
- calculateError.py: Contains helper functions to compute error metrics
- dataHelpers.py: Functions to generate the dataset use in demo.py and for for formatting data.
- demo.py: Trains and evaluates ASATVN on Water_usage dataset.
- Encoder.py: Encoder is made up of Truncated Cauchy self-attention layer, feed forward and Add & Norm.
- evaluate.py: Contains a rudimentary training function to train ASATVN.
- mse_loss.py: Calculates the mean squared error loss
- optimizer.py: Implements Open AI version of Adam algorithm with weight decay fix.
- SATVN.py: Contains the main class for SATVN (Self-Attentive Time-Variant Network).
- train.py: Contains a rudimentary training function to train ASATVN.
- Truncated_Cauchy_self_attention_block.py: Contains Truncated Cauchy self-attention block described in paper.
- Self_attentive_discriminator.py: Contains self-attentive discriminator described in paper.


## Usage

Run the demo.py script to train and evaluate ASATVN model on Water_usage dataset. 

## Requirements

- Python 3.6
- Torch version 1.2.0
- NumPy 1.14.6.
