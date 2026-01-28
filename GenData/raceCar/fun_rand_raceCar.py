import numpy as np

rand_seed = 1234
Sizestep_size = 0.05
Restep_size = 4000
Aoastep_size = 0.01
Aoa2step_size = 0.01
Hcstep_size = 0.001
Hcbstep_size = 0.001
Scstep_size = 0.001
sample_size = 900
np.random.seed(rand_seed)

#  lower Reynolds numbers (50,000<𝑅𝑒<100,000)
# much lower Reynolds numbers, that is, e4<𝑅𝑒<5e4

sample_Size = np.arange(0.2, 0.9+Sizestep_size, step=Sizestep_size) # 0.2 < size < 0.9
Size = np.random.choice(sample_Size, size=sample_size)

sample_Re = np.arange(1e6, 5e6+Restep_size, step=Restep_size) # 1e5 < Re < 5e6
Re = np.random.choice(sample_Re, size=sample_size)

sample_Aoa = np.arange(-10, 0+Aoastep_size, step=Aoastep_size) # -10 < Aoa < 0
Aoa = np.random.choice(sample_Aoa, size=sample_size)

sample_Aoa2 = np.arange(-10, 0+Aoa2step_size, step=Aoa2step_size) # -10 < Aoa < 0
Aoa2 = np.random.choice(sample_Aoa2, size=sample_size)

sample_Hc = np.arange(0.1, 1.1+Hcstep_size, step=Hcstep_size) # 0.1 < h/c < 1.1
dist_Hc = np.random.choice(sample_Hc, size=sample_size)

sample_Hcb = np.arange(0.05, 0.1+Hcbstep_size, step=Hcbstep_size) # 0.05 < h/c < 0.4
dist_Hcb = np.random.choice(sample_Hcb, size=sample_size)

sample_Sc = np.arange(-0.05, 0.5+Scstep_size, step=Scstep_size) # -0.05 < S/c < 0.5
dist_Sc = np.random.choice(sample_Sc, size=sample_size)

np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Size.txt', Size, fmt='%.2f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Re.txt', Re, fmt='%.1f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Aoa.txt', Aoa, fmt='%.2f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Aoa2.txt', Aoa2, fmt='%.2f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Hc.txt', dist_Hc, fmt='%.3f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Hcb.txt', dist_Hcb, fmt='%.3f', newline='\n')
np.savetxt('random_ReAoaHc/seed_'+str(rand_seed)+'_Scb.txt', dist_Sc, fmt='%.3f', newline='\n')
