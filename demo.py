from dataHelpers import generate_data
import numpy as np
import matplotlib.pyplot as plt
from ASATVN import ASATVN
from evaluate import evaluate
from train import train

#Use a fixed seed for repreducible results
np.random.seed(1)

# Generate the dataset Water_usage
data=np.array([
76.83   ,77.74	,80.47	,79.56	,82.28	,100.92	,113.20	,90.92	,86.83	,82.74  ,83.65	,80.92,
83.19	,83.65	,83.65	,83.65	,86.83	,100.47	,91.38	,101.38	,95.92	,88.19	,88.19	,80.47,
80.92	,79.56	,80.92	,88.19	,91.83	,96.38	,97.29	,102.29	,99.10	,92.74	,87.29	,85.47,
91.38	,92.74	,89.56	,88.65	,93.20	,99.56	,109.11	,124.56	,115.47	,96.38	,92.29	,86.83,
87.29	,85.92	,85.92	,88.65	,91.83	,112.29	,101.83	,125.02	,102.74	,95.01	,91.83	,86.38,
87.29	,88.19	,89.10	,89.10	,103.65	,127.75	,125.47	,125.47	,109.11	,100.01	,95.01	,85.01,
86.83	,86.83	,86.83	,86.83	,100.47	,111.38	,105.47	,102.74	,105.01	,96.38	,94.10	,86.83,
92.74	,93.20	,95.47	,96.38	,99.56	,120.47	,123.20	,114.11	,120.93	,102.74	,101.83	,95.47,
100.01	,100.01	,98.20	,100.01	,103.65	,114.56	,134.11	,131.84	,113.65	,107.29	,102.29	,94.56,
97.29	,98.20	,95.47	,100.47	,116.38	,117.29	,140.93	,120.02	,111.38	,108.65	,105.92	,99.10,
101.83	,102.74	,102.74	,105.47	,108.65	,139.57	,110.47	,118.65	,120.02	,109.11	,108.20	,101.38,
106.38	,108.65	,107.74	,105.92	,129.56	,139.11	,125.93	,123.65	,118.65	,110.47	,110.02	,100.47,
104.1	,106.6	,105.5	,107.5	,117.9	,136.3	,156.8	,135.8	,130	,117.5	,115.8	,105.5,
111.6	,113.2	,113.1	,112.5	,120	,147.6	,149.9	,131.2	,134.6	,122.2	,117.7	,106.8,
111.5	,111.3	,109.5	,112.1	,127	,135.9	,150.4	,135.6	,134.9	,124.1	,120.8	,112.8,
117.4	,118.6	,119.2	,119.7	,128.6	,142.8	,170	,145.9	,140.1	,128.7	,123.4	,114.6,
120.2	,122	,121.3	,123.2	,141.1	,129.7	,152.4	,141.9	,137	,129	,124.6	,117.3,
122.7	,121	,122	,122	,126.3	,158.1	,164.9	,143.3	,151.4	,136.8	,133.1	,124.8,
132.6	,130.2	,129.6	,129.7	,133.7	,148.3	,155.1	,157.2	,147.2	,142.7	,135.9	,123.8,
132.3	,132.7	,130.7	,129.9	,145.5	,156.6	,161.7	,156	,146.1	,136.8	,132.5	,129.5,
129.5	,134.7	,136.6	,138.4	,149.6	,159.5	,171.4	,162.1	,163.1	,152.4	,145.5	,133.9,
136.6	,139.4	,141.2	,144.9	,181.4	,187	,211.4	,178.1	,168	,154.4	,150.4	,139.4,
144.7	,143	 ,148.3	,152.7	,173.3	,226.3	,218.2	,184.6	,174.9	,161.4	,161.4	,145.8
])
train_x, train_y, test_x, test_y, valid_x, valid_y, period,mm = generate_data(data,period = 12)

# Model parameters
in_seq_length = 2 * period
out_seq_length = period
hidden_dim = 75
head = 1
M = 2
input_dim = 1
output_dim = 1
learning_rate = 0.001
batch_size = 16
change_dim = 50
dis_alpha = 0.1
d_model = 30

# Initialise model
ASA = ASATVN(in_seq_length=in_seq_length, out_seq_length=out_seq_length, d_model=d_model,input_dim=input_dim,
                        hidden_dim=hidden_dim, h=head, M=M, output_dim=output_dim, batch_size = batch_size,
                        period=period, n_epochs = 1, learning_rate = learning_rate, train_x=train_x, save_file = './asatvn.pt')

# Train the model
training_costs, training_d_costs, validation_costs = train(ASA, train_x, train_y, valid_x, valid_y, restore_session=False)

# Plot the training curves
# plt.figure()
# plt.plot(training_costs)
# plt.plot(validation_costs)

# Evaluate the model
mase,smape, nrmse = evaluate(ASA, test_x, test_y, return_lists=False)

print('MASE:', mase)
print('SMAPE:', smape)


# Generate and plot forecasts for various samples from the test dataset
samples = [0, 12, 23]
predict_start = 24
y_pred = ASA.forecast(test_x[:, :predict_start][:,samples,:],predict_start)
for i in range(len(samples)):
    pred= mm.inverse_transform(y_pred[:, i, 0][:,np.newaxis])
    true = mm.inverse_transform(test_y[predict_start:, :, :][:, samples[i], 0][:,np.newaxis])
    plt.figure()
    plt.plot(np.arange(ASA.in_seq_length, ASA.in_seq_length + ASA.out_seq_length),
             test_y[predict_start:, :, :][:, samples[i], 0],
             '-')
    plt.plot(np.arange(ASA.in_seq_length, ASA.in_seq_length + ASA.out_seq_length),
             y_pred[:, i, 0],
             '-', linewidth=0.7, label='mean')
plt.show()