import numpy as np

rand_seed = 123
step_size = 0.01
sample_size = 28**2
np.random.seed(rand_seed)

sample_S = np.arange(0.5, 2+step_size, step=step_size) # 0.5 < S < 2 
dist_S = np.random.choice(sample_S, size=sample_size)

sample_G = np.arange(-0.2, 0.6+step_size, step=step_size) # -0.4 < G < 0.4
dist_G = np.random.choice(sample_G, size=sample_size)

sample_H = np.arange(0.4, 1+step_size, step=step_size) 
dist_H = np.random.choice(sample_H, size=sample_size)

np.savetxt('random_SGH/seed_'+str(rand_seed)+'_S.txt', dist_S, fmt='%.2f', newline='\n')
np.savetxt('random_SGH/seed_'+str(rand_seed)+'_G.txt', dist_G, fmt='%.2f', newline='\n')
np.savetxt('random_SGH/seed_'+str(rand_seed)+'_H.txt', dist_H, fmt='%.2f', newline='\n')

