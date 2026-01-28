import numpy as np

rand_seed = 100
Restep_size = 4500
Aoastep_size = 0.01
sample_size = 1029
np.random.seed(rand_seed)

#  lower Reynolds numbers (50,000<𝑅𝑒<100,000)
# much lower Reynolds numbers, that is, e4<𝑅𝑒<5e4

#sample_Re = np.arange(1e3, 1e5+Restep_size, step=Restep_size) # 1000 < Re < 100000
#Re = np.random.choice(sample_Re, size=sample_size)
sample_Re = np.arange(1e5, 5e6+Restep_size, step=Restep_size) # 1000 < Re < 100000
Re = np.random.choice(sample_Re, size=sample_size)

sample_Aoa = np.arange(-5, 7+Aoastep_size, step=Aoastep_size) # -5 < Aoa < 10
Aoa = np.random.choice(sample_Aoa, size=sample_size)

np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Re.txt', Re, fmt='%.1f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Aoa.txt', Aoa, fmt='%.2f', newline='\n')

