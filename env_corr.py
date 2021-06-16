#env_corr.py
#attempts to quantify the correlation coeff. between real baselines measured over time

from linear_model import *

plt.close('all')

corr_N = 10000
box_n = 100
box_alpha=1
day_thresh = 1.9 #median
winter_i = np.arange(N_winter)
summer_i = np.arange(N_winter, N_full)
wf_i = np.arange(N_full)

day_i   = np.where(env_data[:,0] >= day_thresh)[0]
night_i = np.where(env_data[:,0] <  day_thresh)[0]

names = ("All", "Day", "Night")
sets = (wf_i,day_i, night_i)
colors = ('C2', 'C1', 'C0')
loops = 1

for i,(name,set,color) in enumerate(zip(names,sets,colors)):
    plt.figure(i)
    plt.title(name+" Correlations")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    means = []
    for j in range(loops):        
        random.shuffle(set)
        corrs = np.corrcoef(wfs[set[:corr_N]])
        corrs = corrs[np.tril_indices(corr_N,-1)] #get all values under the diagonal
        n = len(corrs)
        plt.hist(corrs, box_n, weights=np.ones(n)/n, alpha=box_alpha, color=color)
        means.append(np.mean(corrs))
    m = np.mean(means)
    ylims = plt.gca().get_ylim()
    l, = plt.plot([m, m], [ylims[0], ylims[1]], color='red')
    plt.legend([l], ['Mean = {:.3f}'.format(m)],loc='upper left')
    plt.gca().set_ylim(ylims)

plt.show()
    