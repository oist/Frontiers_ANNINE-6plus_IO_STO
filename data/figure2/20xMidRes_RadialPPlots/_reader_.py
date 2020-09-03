from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

plt.figure(figsize=(15, 3))
ax1 = plt.subplot(1,10,1)
ax2 = plt.subplot(1,10,2, sharex = ax1, sharey = ax1)
ax3 = plt.subplot(1,10,3, sharex = ax1, sharey = ax1)
ax4 = plt.subplot(1,10,4)
ax5 = plt.subplot(1,10,5, sharex=ax4, sharey=ax4)
ax6 = plt.subplot(1,10,6, sharex=ax4, sharey=ax4)
ax7 = plt.subplot(1,10,7)
ax8 = plt.subplot(1,10,8)
ax9 = plt.subplot(1,10,9, sharex=ax8, sharey=ax8)
ax10 = plt.subplot(1,10,10, sharex=ax8, sharey=ax8)

FILE_ID = [] 
PATHS = load_directory_content__()
ALL_DAPI_AVG = []
ALL_D0_1PCT_DI4_AVG = []
ALL_D0_3PCT_DI4_AVG = []
ALL_D1_1PCT_DI4_AVG = []
ALL_D1_3PCT_DI4_AVG = []
ALL_D7_1PCT_DI4_AVG = []
ALL_D7_3PCT_DI4_AVG = []

INDIVIDUAL_D0_INSIDE = []
INDIVIDUAL_D0_OUTSIDE = []
INDIVIDUAL_D0_NUCLEUS = []
INDIVIDUAL_D1_INSIDE = []
INDIVIDUAL_D1_OUTSIDE = []
INDIVIDUAL_D1_NUCLEUS = []
INDIVIDUAL_D7_INSIDE = []
INDIVIDUAL_D7_OUTSIDE = []
INDIVIDUAL_D7_NUCLEUS = []

INDIVIDUAL_D0_CONDITION = []
INDIVIDUAL_D1_CONDITION = []
INDIVIDUAL_D7_CONDITION = []

LOWER_BOUND =4
UPPER_BOUND =13

for i in range(len(PATHS[0])):
    file = PATHS[0][i]
    
    if ('KD' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        DI4_DIST_VALUES= np.array(a[1])
        DI4_INTENSITY_VALUES = np.array(a[2])
        DAPI_DIST_VALUES =  np.array(a[3])
        DAPI_INTENSITY_VALUES=  np.array(a[4])
        SOMA_EPI_DIST =  np.array(a[5][0])
        FILE_ID.append(file)
        
        DAPI_SUB = [(DAPI_INTENSITY_VALUES[i]-np.nanmin(DAPI_INTENSITY_VALUES[i])) for i in range(len(DAPI_INTENSITY_VALUES))]
        DAPI_AVG = [(DAPI_SUB[i]/np.nanmax(DAPI_SUB[i])) for i in range(len(DAPI_SUB))]
        if len (np.nanmean(DAPI_AVG, axis=0))==50:
            ALL_DAPI_AVG.append(np.nanmean(DAPI_AVG, axis=0))
        
        SUB = [(DI4_INTENSITY_VALUES[i])-np.nanmin(DI4_INTENSITY_VALUES[i]) for i in range(len(SOMA_EPI_DIST))]
        AVG = [(SUB[i]/np.nanmax(SUB[i])) for i in range(len(SOMA_EPI_DIST)) if 10<SOMA_EPI_DIST[i]<200]#if 25<SOMA_EPI_DIST[i]<150]
       
        if len(AVG)>1:
            print(file)
            if ('D1P' in file)==True:
                MEAN = np.nanmean(AVG, axis=0)
                SEM = sp.stats.sem(AVG, axis=0)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5<DI4_DIST_VALUES[0][k]<10], axis=0)
                #ax4.scatter(1, np.nanmean(temp_), s=10, color='orange')
                INDIVIDUAL_D1_INSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 10<DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(1, np.nanmean(temp_), s=10, color='black')
                INDIVIDUAL_D1_OUTSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5>DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(1, np.nanmean(temp_), s=10, color='blue')
                INDIVIDUAL_D1_NUCLEUS.append(temp_)
                
                
                INTERNALIZED_DYE = DI4_DIST_VALUES[0][MEAN.tolist().index(np.nanmax(MEAN))]
                if ('1pct' in file)==True:
                    INDIVIDUAL_D1_CONDITION .append(np.linspace(1,1,len(temp_)))
                    ax2.plot(DI4_DIST_VALUES[0], MEAN, color='blue')
                    ax2.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='blue', alpha=0.1)
                    #ax2.plot(DI4_DIST_VALUES[0], np.nanmean(DAPI_AVG, axis=0), color='black')
                    #ax5.scatter(np.linspace(1,1, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='blue')
                    ALL_D1_1PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
                    BIN_NORMED = np.nanmean([DI4_INTENSITY_VALUES[k] for k in range(len(SOMA_EPI_DIST)) if 0<SOMA_EPI_DIST[k]<100])
                else:
                    INDIVIDUAL_D1_CONDITION .append(np.linspace(3,3,len(temp_)))
                    ax2.plot(DI4_DIST_VALUES[0], np.nanmean(AVG, axis=0), color='red')
                    ax2.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='red', alpha=0.1)
                    #ax2.plot(DI4_DIST_VALUES[0], np.nanmean(DAPI_AVG, axis=0), color='black')
                    #ax5.scatter(np.linspace(1,1, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='red')
                    ALL_D1_3PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
            elif ('D0P' in file)==True:
                MEAN = np.nanmean(AVG, axis=0)
                SEM = sp.stats.sem(AVG, axis=0)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5<DI4_DIST_VALUES[0][k]<10], axis=0)
                #ax4.scatter(0, np.nanmean(temp_), s=10, color='orange')
                INDIVIDUAL_D0_INSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 10<DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(0, np.nanmean(temp_), s=10, color='black')
                INDIVIDUAL_D0_OUTSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5>DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(0, np.nanmean(temp_), s=10, color='blue')
                INDIVIDUAL_D0_NUCLEUS.append(temp_)                
                
                INTERNALIZED_DYE =  DI4_DIST_VALUES[0][MEAN.tolist().index(np.nanmax(MEAN))]
                
                if ('1pct' in file)==True:
                    INDIVIDUAL_D0_CONDITION .append(np.linspace(1,1,len(temp_)))
                    ax1.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='blue', alpha=0.1)
                    ax1.plot(DI4_DIST_VALUES[0], np.nanmean(AVG, axis=0), color='blue')
                    #ax1.plot(DI4_DIST_VALUES[0], np.nanmean(DAPI_AVG, axis=0), color='black')
                    #ax5.scatter(np.linspace(0,0, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='blue')
                    ALL_D0_1PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
                    BIN_NORMED = np.nanmean([DI4_INTENSITY_VALUES[k] for k in range(len(SOMA_EPI_DIST)) if 0<SOMA_EPI_DIST[k]<100])
                    
                else:
                    INDIVIDUAL_D0_CONDITION .append(np.linspace(3,3,len(temp_)))
                    ax1.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='red', alpha=0.1)
                    ax1.plot(DI4_DIST_VALUES[0], np.nanmean(AVG, axis=0), color='red')
                    #ax1.plot(DI4_DIST_VALUES[0], np.nanmean(DAPI_AVG, axis=0), color='black')
                    #ax5.scatter(np.linspace(0,0, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='red')
                    ALL_D0_3PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
            elif ('D7P' in file)==True:
                MEAN = np.nanmean(AVG, axis=0)
                SEM = sp.stats.sem(AVG, axis=0)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5<DI4_DIST_VALUES[0][k]<10], axis=0)
                #ax4.scatter(2, np.nanmean(temp_), s=10, color='orange')
                INDIVIDUAL_D7_INSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 10<DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(2, np.nanmean(temp_), s=10, color='black')
                INDIVIDUAL_D7_OUTSIDE.append(temp_)
                temp_ = np.nanmean([np.array(AVG)[:,k] for k in range(len(AVG[0])) if 3.5>DI4_DIST_VALUES[0][k]], axis=0)
                #ax4.scatter(2, np.nanmean(temp_), s=10, color='blue')
                INDIVIDUAL_D7_NUCLEUS.append(temp_)
                INTERNALIZED_DYE = DI4_DIST_VALUES[0][MEAN.tolist().index(np.nanmax(MEAN))]
                if ('1pct' in file)==True:
                    INDIVIDUAL_D7_CONDITION .append(np.linspace(1,1,len(temp_)))
                    ax3.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='blue', alpha=0.1)
                    ax3.plot(DI4_DIST_VALUES[0], np.nanmean(AVG, axis=0), color='blue')
                    #ax5.scatter(np.linspace(7,7, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='blue')
                    ALL_D7_1PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
                    BIN_NORMED = np.nanmean([DI4_INTENSITY_VALUES[k] for k in range(len(SOMA_EPI_DIST)) if 0<SOMA_EPI_DIST[k]<100])
                else:
                    INDIVIDUAL_D7_CONDITION .append(np.linspace(3,3,len(temp_)))
                    ax3.fill_between(DI4_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, color='red', alpha=0.1)
                    ax3.plot(DI4_DIST_VALUES[0], np.nanmean(AVG, axis=0), color='red')
                    #ax5.scatter(np.linspace(7,7, len(INTERNALIZED_DYE)),INTERNALIZED_DYE, color='red')
                    ALL_D7_3PCT_DI4_AVG.append(np.nanmean(INTERNALIZED_DYE))
    elif ('BB_SPARSEGFP' in file)==True and ('pyobj' in file)==True:
        a = load_pickle_file(file)
        GFP_DIST = np.array(a[1])
        GFP_INTENSITY = np.array(a[2])
        
        GFP_SUB = [(GFP_INTENSITY[i]-np.nanmin(GFP_INTENSITY[i])) for i in range(len(GFP_INTENSITY))]
        GFP_AVG = [(GFP_SUB[i]/np.nanmax(GFP_SUB[i])) for i in range(len(GFP_SUB))]
        SEM = sp.stats.sem(GFP_AVG, axis=0)
        MEAN = np.nanmean(GFP_AVG, axis=0)
        ax1.plot(GFP_DIST[0], np.nanmean(GFP_AVG, axis=0), color='green')
        ax2.plot(GFP_DIST[0], np.nanmean(GFP_AVG, axis=0), color='green')
        ax3.plot(GFP_DIST[0], np.nanmean(GFP_AVG, axis=0), color='green')
        ax1.fill_between(GFP_DIST[0], MEAN+SEM, MEAN-SEM, color='green', alpha=0.1)
        ax2.fill_between(GFP_DIST[0], MEAN+SEM, MEAN-SEM, color='green', alpha=0.1)
        ax3.fill_between(GFP_DIST[0], MEAN+SEM, MEAN-SEM, color='green', alpha=0.1)

MEAN = np.nanmean(DAPI_AVG, axis=0)
SEM = sp.stats.sem(ALL_DAPI_AVG, axis=0)
ax1.plot(DAPI_DIST_VALUES[0], MEAN)
ax2.plot(DAPI_DIST_VALUES[0], MEAN)
ax3.plot(DAPI_DIST_VALUES[0], MEAN)
ax1.fill_between(DAPI_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, alpha=0.1)
ax2.fill_between(DAPI_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, alpha=0.1)
ax3.fill_between(DAPI_DIST_VALUES[0], MEAN+SEM, MEAN-SEM, alpha=0.1)

MEAN_NUCLEUS_D0 = np.concatenate(INDIVIDUAL_D0_NUCLEUS)
MEAN_NUCLEUS_D1 = np.concatenate(INDIVIDUAL_D1_NUCLEUS)
MEAN_NUCLEUS_D7 = np.concatenate(INDIVIDUAL_D7_NUCLEUS)
MEAN_INSIDE_D0 = np.concatenate(INDIVIDUAL_D0_INSIDE)-np.concatenate(INDIVIDUAL_D0_NUCLEUS)
MEAN_INSIDE_D1 = np.concatenate(INDIVIDUAL_D1_INSIDE)-np.concatenate(INDIVIDUAL_D1_NUCLEUS)
MEAN_INSIDE_D7 = np.concatenate(INDIVIDUAL_D7_INSIDE)-np.concatenate(INDIVIDUAL_D7_NUCLEUS)
MEAN_OUTSIDE_D0 = np.concatenate(INDIVIDUAL_D0_OUTSIDE) - MEAN_INSIDE_D0
MEAN_OUTSIDE_D1 = np.concatenate(INDIVIDUAL_D1_OUTSIDE) - MEAN_INSIDE_D1
MEAN_OUTSIDE_D7 = np.concatenate(INDIVIDUAL_D7_OUTSIDE) - MEAN_INSIDE_D7


ax4.scatter('D0', np.nanmean(MEAN_INSIDE_D0)/np.nanmean(MEAN_INSIDE_D0), color='orange')
ax4.scatter('D1', np.nanmean(MEAN_INSIDE_D1)/np.nanmean(MEAN_INSIDE_D0), color='orange')
ax4.scatter('D7', np.nanmean(MEAN_INSIDE_D7)/np.nanmean(MEAN_INSIDE_D0), color='orange')

ax5.scatter('D0', np.nanmean(MEAN_NUCLEUS_D0)/np.nanmean(MEAN_NUCLEUS_D0), color='blue')
ax5.scatter('D1', np.nanmean(MEAN_NUCLEUS_D1)/np.nanmean(MEAN_NUCLEUS_D0), color='blue')
ax5.scatter('D7', np.nanmean(MEAN_NUCLEUS_D7)/np.nanmean(MEAN_NUCLEUS_D0), color='blue')

ax6.scatter('D0', np.nanmean(MEAN_OUTSIDE_D0)/np.nanmean(MEAN_OUTSIDE_D0), color='black')
ax6.scatter('D1', np.nanmean(MEAN_OUTSIDE_D1)/np.nanmean(MEAN_OUTSIDE_D0), color='black')
ax6.scatter('D7', np.nanmean(MEAN_OUTSIDE_D7)/np.nanmean(MEAN_OUTSIDE_D0), color='black')

ax4.scatter(['D0' for i in range(len(MEAN_INSIDE_D0))], MEAN_INSIDE_D0/np.nanmean(MEAN_INSIDE_D0), s=3, color='orange', alpha=0.1)
ax4.scatter(['D1' for i in range(len(MEAN_INSIDE_D1))], MEAN_INSIDE_D1/np.nanmean(MEAN_INSIDE_D0), s=3, color='orange', alpha=0.1)
ax4.scatter(['D7' for i in range(len(MEAN_INSIDE_D7))], MEAN_INSIDE_D7/np.nanmean(MEAN_INSIDE_D0), s=3, color='orange', alpha=0.1)
ax5.scatter(['D0' for i in range(len(MEAN_NUCLEUS_D0))], MEAN_NUCLEUS_D0/np.nanmean(MEAN_NUCLEUS_D0), s=3, color='blue', alpha=0.1)
ax5.scatter(['D1' for i in range(len(MEAN_NUCLEUS_D1))], MEAN_NUCLEUS_D1/np.nanmean(MEAN_NUCLEUS_D0), s=3, color='blue', alpha=0.1)
ax5.scatter(['D7' for i in range(len(MEAN_NUCLEUS_D7))], MEAN_NUCLEUS_D7/np.nanmean(MEAN_NUCLEUS_D0), s=3, color='blue', alpha=0.1)
ax6.scatter(['D0' for i in range(len(MEAN_OUTSIDE_D0))], MEAN_OUTSIDE_D0/np.nanmean(MEAN_OUTSIDE_D0), s=3, color='black', alpha=0.1)
ax6.scatter(['D1' for i in range(len(MEAN_OUTSIDE_D1))], MEAN_OUTSIDE_D1/np.nanmean(MEAN_OUTSIDE_D0), s=3, color='black', alpha=0.1)
ax6.scatter(['D7' for i in range(len(MEAN_OUTSIDE_D7))], MEAN_OUTSIDE_D7/np.nanmean(MEAN_OUTSIDE_D0), s=3, color='black', alpha=0.1)

MEAN = np.nanmean(MEAN_NUCLEUS_D0)/np.nanmean(MEAN_NUCLEUS_D0)
SEM = sp.stats.sem(MEAN_NUCLEUS_D0/np.nanmean(MEAN_NUCLEUS_D0))
ax5.plot(('D0', 'D0'), (MEAN+SEM, MEAN-SEM), color='blue')

MEAN = np.nanmean(MEAN_INSIDE_D0)/np.nanmean(MEAN_INSIDE_D0)
SEM = sp.stats.sem(MEAN_INSIDE_D0/np.nanmean(MEAN_INSIDE_D0))
ax4.plot(('D0', 'D0'), (MEAN+SEM, MEAN-SEM), color='orange')

MEAN = np.nanmean(MEAN_OUTSIDE_D0)/np.nanmean(MEAN_OUTSIDE_D0)
SEM = sp.stats.sem(MEAN_OUTSIDE_D0/np.nanmean(MEAN_OUTSIDE_D0))
ax6.plot(('D0', 'D0'), (MEAN+SEM, MEAN-SEM), color='black')

MEAN = np.nanmean(MEAN_NUCLEUS_D1)/np.nanmean(MEAN_NUCLEUS_D0)
SEM = sp.stats.sem(MEAN_NUCLEUS_D1/np.nanmean(MEAN_NUCLEUS_D0))
ax5.plot(('D1', 'D1'), (MEAN+SEM, MEAN-SEM), color='blue')

MEAN = np.nanmean(MEAN_INSIDE_D1)/np.nanmean(MEAN_INSIDE_D0)
SEM = sp.stats.sem(MEAN_INSIDE_D1/np.nanmean(MEAN_INSIDE_D0))
ax4.plot(('D1', 'D1'), (MEAN+SEM, MEAN-SEM), color='orange')

MEAN = np.nanmean(MEAN_OUTSIDE_D1)/np.nanmean(MEAN_OUTSIDE_D0)
SEM = sp.stats.sem(MEAN_OUTSIDE_D1/np.nanmean(MEAN_OUTSIDE_D0))
ax6.plot(('D1', 'D1'), (MEAN+SEM, MEAN-SEM), color='black')

MEAN = np.nanmean(MEAN_NUCLEUS_D7)/np.nanmean(MEAN_NUCLEUS_D0)
SEM = sp.stats.sem(MEAN_NUCLEUS_D7/np.nanmean(MEAN_NUCLEUS_D0))
ax5.plot(('D7', 'D7'), (MEAN+SEM, MEAN-SEM), color='blue')

MEAN = np.nanmean(MEAN_INSIDE_D7)/np.nanmean(MEAN_INSIDE_D0)
SEM = sp.stats.sem(MEAN_INSIDE_D7/np.nanmean(MEAN_INSIDE_D0))
ax4.plot(('D7', 'D7'), (MEAN+SEM, MEAN-SEM), color='orange')

MEAN = np.nanmean(MEAN_OUTSIDE_D7)/np.nanmean(MEAN_OUTSIDE_D0)
SEM = sp.stats.sem(MEAN_OUTSIDE_D7/np.nanmean(MEAN_OUTSIDE_D0))
ax6.plot(('D7', 'D7'), (MEAN+SEM, MEAN-SEM), color='black')

INDIVIDUAL_D0_CONDITION = np.concatenate(INDIVIDUAL_D0_CONDITION)
INDIVIDUAL_D1_CONDITION = np.concatenate(INDIVIDUAL_D1_CONDITION)
INDIVIDUAL_D7_CONDITION = np.concatenate(INDIVIDUAL_D7_CONDITION)

MEAN = [MEAN_INSIDE_D0[i] for i in range(len(INDIVIDUAL_D0_CONDITION)) if INDIVIDUAL_D0_CONDITION[i]==1]
ax7.scatter('D0', np.nanmean(MEAN), color='blue')
MEAN = [MEAN_INSIDE_D1[i] for i in range(len(INDIVIDUAL_D1_CONDITION)) if INDIVIDUAL_D1_CONDITION[i]==1]
ax7.scatter('D1', np.nanmean(MEAN), color='blue')
MEAN = [MEAN_INSIDE_D7[i] for i in range(len(INDIVIDUAL_D7_CONDITION)) if INDIVIDUAL_D7_CONDITION[i]==1]
ax7.scatter('D7', np.nanmean(MEAN), color='blue')

MEAN = [MEAN_INSIDE_D0[i] for i in range(len(INDIVIDUAL_D0_CONDITION)) if INDIVIDUAL_D0_CONDITION[i]==3]
ax7.scatter('D0', np.nanmean(MEAN), color='red')
MEAN = [MEAN_INSIDE_D1[i] for i in range(len(INDIVIDUAL_D1_CONDITION)) if INDIVIDUAL_D1_CONDITION[i]==3]
ax7.scatter('D1', np.nanmean(MEAN), color='red')
MEAN = [MEAN_INSIDE_D7[i] for i in range(len(INDIVIDUAL_D7_CONDITION)) if INDIVIDUAL_D7_CONDITION[i]==3]
ax7.scatter('D7', np.nanmean(MEAN), color='red')
