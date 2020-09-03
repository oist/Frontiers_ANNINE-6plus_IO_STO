# -*- coding: utf-8 -*-

def line_func(x, a, b):
    return a * x + b

def maximum_projection(IM_ARRAY_, image_x_size, image_y_size):
    for i in range(len(IM_ARRAY_)):
        IM = np.nanmax(IM_ARRAY_, axis=1)
        IM= np.reshape(IM, (-1,image_x_size))
    return IM

def threshold_image(input_, threshold_ratio = 0.25, absolute_threshold = 0):
    output_ = []
    if absolute_threshold >0:
        threshold_ = absolute_threshold
    else:
        threshold_ = np.nanmax(input_)*threshold_ratio
    for i in range(len(input_)):
        temp_ = []
        for j in range(len(input_[i])):
            if input_[i][j] >threshold_:
                temp_.append(input_[i][j])
            else:
                temp_.append(np.nan)
        output_.append(temp_)
    return np.array(output_)

def threshold_grayscale_image(IM_ARRAY_, th, rep, sign='pos'):
    thresholded_image = []
    for i in range(len(IM_ARRAY_)):
        if sign=='pos': #sign='pos' TO CUT ABOVE A VALUE
            temp__ = [x if np.float64(x)>th else rep for x in IM_ARRAY_[i]]
        if sign=='neg': #sign='neg' TO CUT BELOW A VALUE
            temp__ = [x if np.float64(x)<th else rep for x in IM_ARRAY_[i]]
        thresholded_image.append(temp__)
    return thresholded_image

def grayscale_to_matrix(input_):
    output_ = []
    for i in range(len(input_)):
        for j in range(len(input_[i])):
            if input_[i][j] > 0:
                output_.append([i, j])
    return output_

def check_cluster_consistency(input_, labels_):
    sil_score = metrics.silhouette_score(input_, labels_, metric='sqeuclidean')
    euc_dist = np.nanmean(sklearn.metrics.pairwise.euclidean_distances(input_))
    return sil_score, euc_dist

def OPTICS_clustering(input_):
    optics = sklearn.cluster.OPTICS(xi=0.001, min_cluster_size=5 )
    svd = sklearn.decomposition.TruncatedSVD(n_components=6)
    of = optics.fit(np.array(input_))
    return of

def remove_trend_array_2(array, function): #FITS AND SUBTRACTS function FOR EACH ELEMENT OF LIST
    corrected_array = []
    for i in range(len(array)):
        x_data = np.linspace(0, len(array[i]), len(array[i]))
        popt, pcov = curve_fit(function, x_data, array[i])
        corrected_array.append(array[i] - line_func(x_data, *popt))
    return corrected_array

def average_frames(IM_ARRAY_, start__, stop__):
    AVG_ARRAY_ = []
    for i in np.arange(start__, stop__):
        AVG_ARRAY_.append(IM_ARRAY_[i])
    return np.nanmean(AVG_ARRAY_, axis=0)

def crop_image(array, w, x0, x1, y0, y1):
    '''
    w = source image width
    x0 = left bound
    x1 = right bound
    y0 = top bound
    y1 = bottom bound
    '''
    cropped_image = []
    for i in range(len(array)):
        temp = np.reshape(array[i], (-1,w))
        to_append = []
        for j in range(len(temp)):
            if y0<j<y1:
                to_append.append(temp[j][x0:x1])
        cropped_image.append(np.concatenate(to_append))
    return np.array(cropped_image), x1-x0

def get_frequency_power_from_pixels(IM_ARRAY_, low, high, image_x_size, image_y_size, sf = 80):
    IM_POWER__ = []
    for i in range(len(IM_ARRAY_)):
        data = IM_ARRAY_[i]
        time = np.arange(data.size) / sf
        win = 4 * sf
        freqs, psd = signal.welch(data, sf, nperseg=win, scaling='spectrum')
        idx_delta = np.logical_and(freqs >= low, freqs <= high)
        freq_res = freqs[1] - freqs[0] 
        delta_power = simps(psd[idx_delta], dx=freq_res)
        IM_POWER__.append(delta_power)
    IM = np.reshape(IM_POWER__, (-1,image_x_size))
    return IM

def mean_confidence_interval(data, confidence=0.95):
    data = 1.0 * np.array(data)
    data_len = len(data)
    m, se = np.nanmean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., data_len-1)
    return m, h, h


def preprocess_image(PATH, denoise_method__):
    IM_ARRAY_ = []
    
    print('1-IMPORTING DATA')
    try:
        dataset = Image.open(PATH)
        w,h = dataset.size
        tifarray = np.zeros((w,h,dataset.n_frames))
        print('//file importation -- ok')
        print(PATH)
    except:
        print('//could not import file')
        
    try:
        for i in range(dataset.n_frames):
            dataset.seek(i)
            a = dataset
            IM_ARRAY_.append(np.concatenate(np.array(a)))
        SAMPLING_FREQUENCY = len(IM_ARRAY_) / RECORDING_LENGTH 
        IM_ARRAY_=np.array(IM_ARRAY_).transpose()
        ORIGINAL = IM_ARRAY_
        print('//image array merged -- ok')
    except:
        print('//could not merge array')
    print('done')
    
    print('2-PREPROCESSING DATA')
    if True:
        print('//croping')
        if ('crop with IRed ROI' in options__)==True:
            PATH = load_specific_file__()
            print('//using ROI crop')
            ROI_crop = pd.read_csv(PATH)
            EMPTY_ARRAY = np.empty([len(IM_ARRAY_),len(IM_ARRAY_[0])])
            for k in range(len(ROI_crop.X)):
                EMPTY_ARRAY[ROI_crop.Y[k]*w+ROI_crop.X[k]] = IM_ARRAY_[ROI_crop.Y[k]*w+ROI_crop.X[k]]
            IM_ARRAY_ = EMPTY_ARRAY
            
        IM_ARRAY_, w = crop_image(np.transpose(IM_ARRAY_), w, x0=80, x1=180, y0=70, y1=170)
        IM_ARRAY_=np.array(IM_ARRAY_).transpose()
        
        print('//normalizing')
        SCALE_FIT = sklearn.preprocessing.Normalizer().fit(IM_ARRAY_.transpose())
        IM_ARRAY_ = SCALE_FIT.transform(IM_ARRAY_.transpose()).transpose()
        
        if denoise_method__=='nlmeans' or ('calculate intensity epicenter' in  options__)==True:
            print('//denoising -- nlmeans')
            ORIGINAL__ = []
            for i in range(len(IM_ARRAY_.transpose())):
                ORIGINAL__.append(np.reshape(IM_ARRAY_[:,i], (-1,w)))
            ORIGINAL__ = np.array(ORIGINAL__) 
            sigma_est = np.mean(estimate_sigma(IM_ARRAY_, multichannel=True))
            patch_kw = dict(patch_size=5, patch_distance = 2, multichannel=True)
            Y = denoise_nl_means(ORIGINAL__, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
            AVERAGE_IMAGE = np.reshape(np.nanmean([np.concatenate(Y[i]) for i in range(len(Y))], axis=0), (-1,w))
            
            plt.figure(num='AVG'+PATH)
            plt.imshow(AVERAGE_IMAGE, interpolation='hamming')
            
            if denoise_method__=='nlmeans' :
                IM_ARRAY_ = []
                for i in range(len(Y)):
                    IM_ARRAY_.append(np.concatenate(Y[i]))
                IM_ARRAY_=np.array(IM_ARRAY_).transpose()
        if  ('calculate intensity epicenter' in  options__)==True:
            image_peak_coordinates = skimage.feature.peak.peak_local_max(AVERAGE_IMAGE, num_peaks=1)
            
    print('//removing trend')  
    IM_ARRAY_ = remove_trend_array_2(np.array(IM_ARRAY_), line_func)
    print('done')
    return IM_ARRAY_, w, h, AVERAGE_IMAGE, SAMPLING_FREQUENCY, image_peak_coordinates

def Image_Decomposition(IM_ARRAY_, w, h, AVERAGE_IMAGE, LOW_FREQ, HIGH_FREQ, SAMPLING_FREQUENCY, IMG_NUM, RECORDING_LENGTH, bin_size, method__, denoise_method__, clustering_method_, residuals__, options__):
    ROI_decomposition = []
    FREQUENCY_COMPONENTS = []
    DecompositionResults = []
    ID_ = []
    AverageClusterTimecourse = []
    PixelsPerCluster = []

    if method__  == 'partial':
        plt.figure(figsize=(IMG_NUM*2,3), num=PATH.split('\\')[-1])
    elif method__ == 'full':
        plt.figure(figsize=(IMG_NUM*2,9), num=PATH.split('\\')[-1])
    
    FREQUENCIES = np.linspace(LOW_FREQ, HIGH_FREQ, IMG_NUM+1)
    if (method__ in ['partial', 'full'])==True:
        for i in range(IMG_NUM):
            if method__  == 'partial':
                ax=plt.subplot(1,IMG_NUM,i+1)
            elif method__ == 'full':
                ax=plt.subplot(4,IMG_NUM,i+1)
        
            print('//decomposing frequencies')
            IM = get_frequency_power_from_pixels(IM_ARRAY_, FREQUENCIES[i], FREQUENCIES[i+1], w, h, sf = SAMPLING_FREQUENCY)
            FREQUENCY_COMPONENTS.append(IM)
            ax.imshow(IM, interpolation='hamming', cmap=cm.jet)
            ax.set_title(str( FREQUENCIES[i])+'-'+str(FREQUENCIES[i+1]))
            print('done')
            

            FREQ__ .append(FREQUENCIES[i])
        
            if method__ == 'full':    
                print('Pass'+str(i+1)+ ' '+str(FREQ__[-1])+'Hz')
                print('Preparing clustering --')
                print('//extracting pixels')
                
                absolute_threshold_ = np.percentile(IM, residuals__)
                thresholded_frequency_component = threshold_grayscale_image(IM, absolute_threshold_, 0, 'pos')
                cluster_centers_ = np.array(np.where(np.array(thresholded_frequency_component)>0)).transpose()
                
                print('//removing noisy pixels')

                absolute_threshold_ = np.percentile(np.concatenate(AVERAGE_IMAGE), residuals__)
                z_score = 3*np.std(AVERAGE_IMAGE)+np.min(AVERAGE_IMAGE) 
                noisy_pixels = threshold_grayscale_image(AVERAGE_IMAGE, absolute_threshold_, 0, 'neg')
                noisy_pixels_centers = np.array(np.where(np.array(noisy_pixels)>0)).transpose()
                
                #FOLLOWING WILL EXCLUDE PIXELS THAT DO NOT PASS Z_SCORE AND PSD-THRESHOLD
                temp__ = []
                for x in cluster_centers_:
                    exclude_ = False
                    for y in noisy_pixels_centers:
                        if x[0]==y[0] and x[1]==y[1]:
                            exclude_ = True
                    if exclude_==False:
                        temp__.append(x)
                cluster_centers_ = np.array(temp__)

                
                print('done')
                
                IM_ARRAY__freq = []
                for j in range(len(cluster_centers_)):
                    idx = cluster_centers_[j][0]*w+cluster_centers_[j][1]
                    IM_ARRAY__freq.append(IM_ARRAY_[int(idx)])
                
                if clustering_method_ == 'SVD-OPTICS':
                    ax2=plt.subplot(4,IMG_NUM,i+1+IMG_NUM)
                    CLUSTERING_RESULTS = OPTICS_clustering(IM_ARRAY__freq)
                    KMC_CLUST = CLUSTERING_RESULTS.labels_
                    N_CLUSTERS = np.max(KMC_CLUST)
                    clr = cm.viridis(np.linspace(0, 1, N_CLUSTERS+1)).tolist()
                    for k in range(len(CLUSTERING_RESULTS.ordering_)):
                        ax2.scatter(CLUSTERING_RESULTS.ordering_[k], CLUSTERING_RESULTS.reachability_[k], c=np.array([clr[CLUSTERING_RESULTS.labels_[k]]]))

              
                ax3=plt.subplot(4,IMG_NUM,(i+1+int(IMG_NUM*2)))

                try:
                    IM_ARRAY_BOOTSTRAP = np.nanmedian([IM_ARRAY_[bootstrap_ID[i]] for i in range(len(bootstrap_ID))])
                    ax3.plot(np.linspace(0, RECORDING_LENGTH, len(IM_ARRAY__freq[0])), np.nanmean(temp__, axis=0), c='red')
                    AverageClusterTimecourse.append(np.nanmean(temp__, axis=0))
                except:
                    print('no noise cluster')
                
                for j in range(N_CLUSTERS):
                    if True:
                        temp__ = [IM_ARRAY__freq[k] for k in range(len(IM_ARRAY__freq)) if KMC_CLUST[k]==j]
                        ax3.plot(np.linspace(0, RECORDING_LENGTH, len(IM_ARRAY__freq[0])), np.nanmean(temp__, axis=0), c=clr[j])
                        AverageClusterTimecourse.append(np.nanmean(temp__, axis=0))
                        PixelsPerCluster.append(len(temp__))
                    
                try:
                    temp__ = [IM_ARRAY__freq[k] for k in range(len(IM_ARRAY__freq)) if KMC_CLUST[k]==-1]
                    ax3.plot(np.linspace(0, RECORDING_LENGTH, len(IM_ARRAY__freq[0])), np.nanmean(temp__, axis=0), c='red')
                    AverageClusterTimecourse.append(np.nanmean(temp__, axis=0))
                except:
                    pass
                
                temp__ = [IM_ARRAY__freq[k] for k in range(len(IM_ARRAY__freq))]
                MEAN = np.nanmean(temp__, axis=0)
                SEM = sp.stats.sem(temp__, axis=0, nan_policy='omit')
                ax3.plot(np.linspace(0, RECORDING_LENGTH, len(IM_ARRAY__freq[0])), MEAN, c='black')
                ax3.fill_between(np.linspace(0, RECORDING_LENGTH, len(IM_ARRAY__freq[0])), MEAN+SEM, MEAN-SEM, color='black', alpha=0.1)
                
                AVERAGE_FreqDec_Pixels = temp__ 
        
                ax4=plt.subplot(4,IMG_NUM,(i+1+int(IMG_NUM*3)), sharex=ax, sharey=ax)
                #REATTRIBUTE OSCLILLATING SOURCES
                oscillation_power = []
                for j in range(len(KMC_CLUST)):
                    try:
                        ax4.scatter(cluster_centers_[j][1], cluster_centers_[j][0], c=np.array([clr[KMC_CLUST[j]]]))
                    except:
                        pass
                for j in range(N_CLUSTERS):
                    if KMC_CLUST[j]!=-1:
                        temp__ = [cluster_centers_[:,0][k] for k in range(len(KMC_CLUST)) if KMC_CLUST[k]==j]
                        temp__2 = [cluster_centers_[:,1][k] for k in range(len(KMC_CLUST)) if KMC_CLUST[k]==j]
                        oscillation_power = [cluster_centers_[:,1][k] for k in range(len(KMC_CLUST)) if KMC_CLUST[k]==j]
                    try:
                        for k in range(len(temp__)):
                            oscillation_power.append(IM[temp__[k]][temp__2[k]])
                    except:
                        pass
                    if len(temp__)>=3:
                        try:
                            trg = matplotlib.tri.Triangulation(temp__2, temp__)
        
                        except:
                            pass
                        
                ax4.matshow(IM, cmap=cm.gray_r, alpha=0.5)
                ID = PATH.split('\\')[-1]
                
                if ('save analysis' in options__) == True:
                    try:
                        if (ID in ID_)==False:
                            second_map_cluster_centers_, silscore_2 = check_cluster_consistency(cluster_centers_, KMC_CLUST)
                            if easygui.boolbox(msg = 'Take/discard ?')==True:
                                clusters_to_save = np.array(easygui.multchoicebox(msg = 'Select epi-clusters to save', choices=np.linspace(0, N_CLUSTERS-1, N_CLUSTERS)))
                                for i in range(len(clusters_to_save)):
                                    cluster_id = np.int(np.float(clusters_to_save[i]))
                                    DecompositionResults.append([ID, FREQ__[-1],  second_map_cluster_centers_, silscore_2, PixelsPerCluster, AverageClusterTimecourse[cluster_id]])
                                    ID_.append(ID)
                        else:
                            print('ENTRY IS ALREADY IN DB //')
                    except:
                        pass
                        
    return IM, AVERAGE_FreqDec_Pixels, cluster_centers_, KMC_CLUST, CLUSTERING_RESULTS
