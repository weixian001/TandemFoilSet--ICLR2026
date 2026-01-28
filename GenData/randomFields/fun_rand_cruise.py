import numpy as np

rand_seed = 900 #784
step_size = 0.01
Restep_size = 1000
Aoastep_size = 0.01
Aoa2step_size = 0.01
sample_size = 30**2
np.random.seed(rand_seed)

sample_S = np.arange(0.5, 2+step_size, step=step_size) # 0.5 < S < 2 
dist_S = np.random.choice(sample_S, size=sample_size)

sample_G = np.arange(-0.8, 0.8+step_size, step=step_size) # -0.8 < G < 0.8
dist_G = np.random.choice(sample_G, size=sample_size)

sample_Re = np.arange(1e5, 5e6+Restep_size, step=Restep_size) # 1e5 < Re < 5e6
Re = np.random.choice(sample_Re, size=sample_size)

sample_Aoa = np.arange(-5, 6+Aoastep_size, step=Aoastep_size) # -5 < Aoa < 6
Aoa = np.random.choice(sample_Aoa, size=sample_size)

sample_Aoa2 = np.arange(-5, 6+Aoa2step_size, step=Aoa2step_size) # -5 < Aoa < 6
Aoab = np.random.choice(sample_Aoa2, size=sample_size)

np.savetxt('random_cruise/seed_'+str(rand_seed)+'_S.txt', dist_S, fmt='%.2f', newline='\n')
np.savetxt('random_cruise/seed_'+str(rand_seed)+'_G.txt', dist_G, fmt='%.2f', newline='\n')
np.savetxt('random_cruise/seed_'+str(rand_seed)+'_Re.txt', Re, fmt='%.1f', newline='\n')
np.savetxt('random_cruise/seed_'+str(rand_seed)+'_Aoa.txt', Aoa, fmt='%.2f', newline='\n')
np.savetxt('random_cruise/seed_'+str(rand_seed)+'_Aoab.txt', Aoab, fmt='%.2f', newline='\n')
