import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('frnn_trt_bs_latency.csv')
#print(data.columns)
bs = data['Batch size']
latency = data['Latency (ms)']

fig, ax = plt.subplots()

ax.plot(bs, latency, '-o')
ax.set_xlabel('batch size')
ax.set_ylabel('e2e latency (ms)')
ax.set_title('128ms subsequence 2-layer LSTM')
fig.savefig("a100_latency.pdf")
