import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

'''
def nploadtxt_skiprows(filename, rows_to_skip):
    with open(filename, 'r') as f:
        lines=[]
        for i in range(len(f)):
            if (i in rows_to_skip):
                continue
            lines.append()
        arr = np.genfromtxt(lines)
'''

def gaussian(x, a, mu, s):
    return a * np.exp(-.5*((x-mu)/s)**2) / (s*np.sqrt(2.*np.pi)) 

folder = '/home/liam/Downloads/CuSc_Repeated/Vertical/'

background = np.genfromtxt(folder+'bkg.txt', delimiter='\t', skip_header=17, skip_footer=1)

data = []
for i in range(0,360,2):
    temp_array = np.genfromtxt(folder+str(i)+'.txt', delimiter='\t', skip_header=17, skip_footer=1)
    temp_array[:,1] -= background[:,1]
    data.append(temp_array)

data = np.array(data)

fit_data = []
for i in range(data.shape[0]):
    popt, pcov = curve_fit(gaussian, data[i,:,0], data[i,:,1], p0=[750.,515.,10.])#, bounds=[[0.,510.,3.], [1600.,520.,15.]])
    fit_data.append(popt)

fit_data = np.array(fit_data)

theta = np.linspace(0.,360.,180, endpoint=False) * (np.pi/180.)
r = fit_data[:,0]*fit_data[:,2]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, '.')
ax.grid(True)

plt.show()

check = input("Save data points? (y/n)")
if (check == 'y' or check == 'Y'):
    import os
    print("Saved as: "+os.getcwd()+'/processed_data.tsv')
    np.savetxt(os.getcwd()+'/processed_data.tsv', list(zip(theta*180./np.pi, r)), delimiter='\t', header='Angle (degrees)\tAmplitude (counts*nm)')
