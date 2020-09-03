from matplotlib import pyplot as plt
from file_functions import *
import numpy as np
import scipy as sp

PATHS = load_directory_content__()

FULL_D0_INTENSITY = []
FULL_D1_INTENSITY = []
FULL_D7_INTENSITY = []
NUCLEUS_D0_ABSOLUTE_INTENSITY = []
SOMA_D0_ABSOLUTE_INTENSITY = []
BACKGROUND_D0_ABSOLUTE_INTENSITY = []
MEMBRANE_D0_RELATIVE_INTENSITY = []

NUCLEUS_D1_ABSOLUTE_INTENSITY = []
SOMA_D1_ABSOLUTE_INTENSITY = []
BACKGROUND_D1_ABSOLUTE_INTENSITY = []
MEMBRANE_D1_RELATIVE_INTENSITY = []

NUCLEUS_D7_ABSOLUTE_INTENSITY = []
SOMA_D7_ABSOLUTE_INTENSITY = []
BACKGROUND_D7_ABSOLUTE_INTENSITY = []
MEMBRANE_D7_RELATIVE_INTENSITY = []

plt.figure(figsize=(3,8))
ax = plt.subplot(311)
ax2 = plt.subplot(312, sharex= ax, sharey=ax)
ax3 = plt.subplot(313, sharex= ax, sharey=ax)

for i in range(len(PATHS[0])):
    file = PATHS[0][i]
    
    if ('D0PIJ' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(a[1])
        DI4_INTENSITY_VALUES = np.array(a[2])
        
        NUCLEUS_D0_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5>DI4_DIST_VALUES[j]], axis=0))
        SOMA_D0_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5<DI4_DIST_VALUES[j]<=10], axis=0))
        BACKGROUND_D0_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if DI4_DIST_VALUES[j]>10], axis=0))

        SUB = [(DI4_INTENSITY_VALUES[j]-np.nanmin(DI4_INTENSITY_VALUES[j])) for j in range(len(DI4_INTENSITY_VALUES))]
        AVG = [(SUB[j]/np.nanmean(SUB[j][-3:-1])) for j in range(len(DI4_INTENSITY_VALUES)) ]
        MEMBRANE_D0_RELATIVE_INTENSITY.append(np.nanmax([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 12>DI4_DIST_VALUES[j]>8], axis=0))

        for j in range(len(AVG)):
            FULL_D0_INTENSITY.append(AVG[j])
        MEAN = np.nanmean(AVG, axis=0)
        SEM = sp.stats.sem(AVG, axis=0)
        ax.plot(DI4_DIST_VALUES, MEAN, color='black', lw=0.7)
        ax.fill_between(DI4_DIST_VALUES, MEAN+SEM, MEAN-SEM, color='blue', alpha=0.2)
        #for j in range(len(AVG)):
            #ax.plot(DI4_DIST_VALUES, AVG[j], color='black', lw=0.7)
    elif ('D1PIJ' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(a[1])
        DI4_INTENSITY_VALUES = np.array(a[2])
        
        NUCLEUS_D1_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5>DI4_DIST_VALUES[j]], axis=0))
        SOMA_D1_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5<DI4_DIST_VALUES[j]<=10], axis=0))
        BACKGROUND_D1_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if DI4_DIST_VALUES[j]>10], axis=0))

        SUB = [(DI4_INTENSITY_VALUES[j]-np.nanmin(DI4_INTENSITY_VALUES[j])) for j in range(len(DI4_INTENSITY_VALUES))]
        AVG = [(SUB[j]/np.nanmean(SUB[j][-3:-1])) for j in range(len(DI4_INTENSITY_VALUES)) ]
        MEMBRANE_D1_RELATIVE_INTENSITY.append(np.nanmax([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 12>DI4_DIST_VALUES[j]>8], axis=0))

        for j in range(len(AVG)):
            FULL_D1_INTENSITY.append(AVG[j])
        MEAN = np.nanmean(AVG, axis=0)
        SEM = sp.stats.sem(AVG, axis=0)
        ax2.plot(DI4_DIST_VALUES, MEAN, color='black', lw=0.7)
        ax2.fill_between(DI4_DIST_VALUES, MEAN+SEM, MEAN-SEM, color='purple', alpha=0.2)
        #for j in range(len(AVG)):
            #ax2.plot(DI4_DIST_VALUES, AVG[j], color='black', lw=0.7)
    elif ('D7PIJ' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(a[1])
        DI4_INTENSITY_VALUES = np.array(a[2])
        
        NUCLEUS_D7_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5>DI4_DIST_VALUES[j]], axis=0))
        SOMA_D7_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 3.5<DI4_DIST_VALUES[j]<=10], axis=0))
        BACKGROUND_D7_ABSOLUTE_INTENSITY.append(np.nanmean([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if DI4_DIST_VALUES[j]>10], axis=0))

        SUB = [(DI4_INTENSITY_VALUES[j]-np.nanmin(DI4_INTENSITY_VALUES[j])) for j in range(len(DI4_INTENSITY_VALUES))]
        AVG = [(SUB[j]/np.nanmean(SUB[j][-3:-1])) for j in range(len(DI4_INTENSITY_VALUES)) ]
        MEMBRANE_D7_RELATIVE_INTENSITY.append(np.nanmax([DI4_INTENSITY_VALUES[:,j] for j in range(len(DI4_INTENSITY_VALUES[0])) if 12>DI4_DIST_VALUES[j]>8], axis=0))

        for j in range(len(AVG)):
            FULL_D7_INTENSITY.append(AVG[j])
        MEAN = np.nanmean(AVG, axis=0)
        SEM = sp.stats.sem(AVG, axis=0)
        ax3.plot(DI4_DIST_VALUES, MEAN, color='black', lw=0.7)
        ax3.fill_between(DI4_DIST_VALUES, MEAN+SEM, MEAN-SEM, color='orange', alpha=0.2)
        #for j in range(len(AVG)):
            #ax3.plot(DI4_DIST_VALUES, AVG[j], color='black', lw=0.7)
#ax.plot(DI4_DIST_VALUES, np.nanmean(FULL_D0_INTENSITY, axis=0), color='orange')
#ax2.plot(DI4_DIST_VALUES, np.nanmean(FULL_D1_INTENSITY, axis=0), color='orange')
#ax3.plot(DI4_DIST_VALUES, np.nanmean(FULL_D7_INTENSITY, axis=0), color='orange')
ax3.set_xlabel('Distance from cell epicenter (um)')
ax3.set_ylabel('Normalized Di4 intensity')
plt.tight_layout()
