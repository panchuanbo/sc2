import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) <= 1:
	print('data_plotter [filename] [lr_rate]')
	exit(0)

losses = np.loadtxt('losses_' + sys.argv[1] + '.txt')
iterations = range(losses.shape[0])

plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('LR = ' + sys.argv[2])
plt.plot(iterations, losses)
plt.savefig(sys.argv[1] + '.png')