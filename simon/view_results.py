import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


ta = np.load("../hyperdepth/forests/td8_ds4/ta.npy") #ta is groundtruth
es = np.load("../hyperdepth/forests/td8_ds4/es.npy") #es are the results

print(es.shape)
fig, axs = plt.subplots(4, 1)
axs[0].imshow(ta[0, :, :])
axs[1].plot(ta[0, 15, :])
axs[2].plot(np.clip(es[0, 15, :, 0], 10, 20)) # channel 1 is the regression
#axs[3].plot(es[0, 15, :, 1]) # channel 2 is the probability by the RandomForest
axs[3].plot(np.clip(es[0, 15, :, 2], 0, 1)) # channel 3 is the residual
#axs[1].plot(es[0, 15, :, 1])
#axs[1].plot(es[0, 15, :, 2])

plt.show()