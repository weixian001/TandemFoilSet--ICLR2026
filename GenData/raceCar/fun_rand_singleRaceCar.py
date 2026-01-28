import numpy as np

rand_seed = 901
Restep_size = 4500
Aoastep_size = 0.01
Hcstep_size = 0.001
sample_size = 901 #1029
np.random.seed(rand_seed)

#  lower Reynolds numbers (50,000<𝑅𝑒<100,000)
# much lower Reynolds numbers, that is, e4<𝑅𝑒<5e4

sample_Re = np.arange(1e5, 5e6+Restep_size, step=Restep_size) # 1e5 < Re < 5e6
Re = np.random.choice(sample_Re, size=sample_size)

sample_Aoa = np.arange(-10, 0+Aoastep_size, step=Aoastep_size) # -10 < Aoa < 0
Aoa = np.random.choice(sample_Aoa, size=sample_size)

sample_Hc = np.arange(0.1, 1.1+Hcstep_size, step=Hcstep_size) # 0.1 < h/c < 1.1
dist_Hc = np.random.choice(sample_Hc, size=sample_size)

np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Re.txt', Re, fmt='%.1f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Aoa.txt', Aoa, fmt='%.2f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Hc.txt', dist_Hc, fmt='%.3f', newline='\n')

