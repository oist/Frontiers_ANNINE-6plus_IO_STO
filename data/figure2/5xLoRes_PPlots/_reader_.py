import pickle
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

PATHS = load_directory_content__()
temp_D0 = []
temp_D1 = []
temp_D7 = []

plt.figure(figsize=(3,8))
ax = plt.subplot(411)
ax2 = plt.subplot(412, sharex= ax, sharey=ax)
ax3 = plt.subplot(413, sharex= ax, sharey=ax)

for i in range(len(PATHS[0])):
    file = PATHS[0][i]
    
    if ('D0PIJ' in file)==True and ('pyobj' in file)==True:
        LOADED_FILE = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(LOADED_FILE[1])
        DI4_INTENSITY_VALUES = np.array(LOADED_FILE[2])
        AVG = np.nanmean(DI4_INTENSITY_VALUES, axis=1)
        ax.plot(DI4_DIST_VALUES, AVG/np.nanmax(AVG), color='black', linewidth = 0.2)
        temp_D0.append(AVG)
    elif ('D1PIJ' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(LOADED_FILE[1])
        DI4_INTENSITY_VALUES = np.array(LOADED_FILE[2])
        AVG = np.nanmean(DI4_INTENSITY_VALUES, axis=1)
        ax2.plot(DI4_DIST_VALUES, AVG/np.nanmax(AVG), color='black', linewidth = 0.2)
        temp_D1.append(AVG)
    elif ('D7PIJ' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(LOADED_FILE[1])
        DI4_INTENSITY_VALUES = np.array(LOADED_FILE[2])
        AVG = np.nanmean(DI4_INTENSITY_VALUES, axis=1)
        ax3.plot(DI4_DIST_VALUES, AVG/np.nanmax(AVG), color='black', linewidth = 0.2)
        temp_D7.append(AVG)
        
MEAN = np.nanmean(temp_D0, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0))
SEM = sp.stats.sem([temp_D0[i]/np.nanmax(np.nanmean(temp_D0, axis=0)) for i in range(len(temp_D0))], axis=0)
ax.plot(DI4_DIST_VALUES, np.nanmean(temp_D0, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0)), color='black')
ax.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='blue', alpha=0.1)

MEAN = np.nanmean(temp_D1, axis=0)/np.nanmax(np.nanmean(temp_D1, axis=0))
SEM = sp.stats.sem([temp_D1[i]/np.nanmax(np.nanmean(temp_D1, axis=0)) for i in range(len(temp_D1))], axis=0)
ax2.plot(DI4_DIST_VALUES, np.nanmean(temp_D1, axis=0)/np.nanmax(np.nanmean(temp_D1, axis=0)), color='black')
ax2.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='purple', alpha=0.1)

MEAN = np.nanmean(temp_D7, axis=0)/np.nanmax(np.nanmean(temp_D7, axis=0))
SEM = sp.stats.sem([temp_D7[i]/np.nanmax(np.nanmean(temp_D7, axis=0)) for i in range(len(temp_D7))], axis=0)
ax3.plot(DI4_DIST_VALUES, np.nanmean(temp_D7, axis=0)/np.nanmax(np.nanmean(temp_D7, axis=0)), color='black')
ax3.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='orange', alpha=0.1)


ax4 = plt.subplot(414)
MEAN = np.nanmean(temp_D0, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0))
SEM = sp.stats.sem([temp_D0[i]/np.nanmax(np.nanmean(temp_D0, axis=0)) for i in range(len(temp_D0))], axis=0)
ax4.plot(DI4_DIST_VALUES, np.nanmean(temp_D0, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0)), color='black')
ax4.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='blue', alpha=0.1)

MEAN = np.nanmean(temp_D1, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0))
SEM = sp.stats.sem([temp_D1[i]/np.nanmax(np.nanmean(temp_D0, axis=0)) for i in range(len(temp_D1))], axis=0)
ax4.plot(DI4_DIST_VALUES, np.nanmean(temp_D1, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0)), color='black')
ax4.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='purple', alpha=0.1)

MEAN = np.nanmean(temp_D7, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0))
SEM = sp.stats.sem([temp_D7[i]/np.nanmax(np.nanmean(temp_D0, axis=0)) for i in range(len(temp_D7))], axis=0)
ax4.plot(DI4_DIST_VALUES, np.nanmean(temp_D7, axis=0)/np.nanmax(np.nanmean(temp_D0, axis=0)), color='black')
ax4.fill_between(DI4_DIST_VALUES, MEAN+SEM,MEAN-SEM, color='orange', alpha=0.1)


ax.set_xlabel('Distance from injection epicenter (um)')
ax.set_ylabel('Normalized intensity D0')
ax2.set_xlabel('Distance from injection epicenter (um)')
ax2.set_ylabel('Normalized intensity D1')
ax3.set_xlabel('Distance from injection epicenter (um)')
ax3.set_ylabel('Normalized intensity D7')

ax4.set_xlabel('Distance from injection epicenter (um)')
ax4.set_ylabel('Normalized to D0')
plt.tight_layout()
