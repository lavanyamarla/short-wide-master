import pickle
import sys
import scipy
import pandas as pd
import numpy as np
import scipy.stats as stats  
import matplotlib.pyplot as plt

dataset = sys.argv[1]

with open('raw_output/' + dataset + '/d_geodesic.pkl', 'rb') as f:
	d_geo = pickle.load(f)

with open('raw_output/' + dataset + '/d_short_wide.pkl', 'rb') as f:
	d_sw = pickle.load(f)

with open('raw_output/' + dataset + '/d_weighted.pkl', 'rb') as f:
	d_weight = pickle.load(f)

with open('raw_output/' + dataset + '/f__short_wide.pkl', 'rb') as f:
	f_sw = pickle.load(f)

#for i in range(len(d_geo)):
#	print(d_weight[i])
#	print(d_sw[i])
#	print(f_sw[i])
#	print(d_geo[i])
#	print()

s_w = float(sys.argv[2])
s_g = float(sys.argv[3])
s_s = float(sys.argv[4])

# Calculate bins
w_bin = [x*s_w + .01 for x in range(1 + int(max(d_weight)/s_w))]
if sys.argv[5] == 'd':
	s_bin = [x*s_s + .01 for x in range(1 + int(max(d_sw)/s_s))]
else:
	s_bin = [x*s_s + .01 for x in range(1 + int(max(f_sw)/s_s))]
g_bin = [x*s_g + .01 for x in range(1 + int(max(d_geo)/s_g))]

# Generate histogram
w_hist = list(np.histogram(d_weight, w_bin)[0])
w_hist /= sum(w_hist)
g_hist = list(np.histogram(d_geo, g_bin)[0])
g_hist /= sum(g_hist)
if sys.argv[5] == 'd':
	s_hist = list(np.histogram(d_sw, s_bin)[0])
else:
	s_hist = list(np.histogram(f_sw, s_bin)[0])
s_hist /= sum(s_hist)

# Generate survival
w_plot = []
for i in range(len(w_hist)):
        w_plot.append(sum(w_hist[i:]))
w_plot.append(0)

g_plot = []
for i in range(len(g_hist)):
        g_plot.append(sum(g_hist[i:]))
g_plot.append(0)

s_plot = []
for i in range(len(s_hist)):
        s_plot.append(sum(s_hist[i:]))
s_plot.append(0)

# Debugging
#print(len(weighted_distance))
#print(count)

# Generate y coordiantes
y = [(len(d_weight) - x)/len(d_weight) for x in range(len(d_weight))]

# Plot
#plt.plot(w_bin, w_plot, label='Weighted')
#plt.plot(g_bin, g_plot, label='Geodesic')
plt.plot(s_bin, s_plot, label='Short Wide')

#fit_alpha, fit_loc, fit_beta=stats.gamma.fit(s_plot)
#x_gamma = np.linspace (0, 12, 100)
#y_gamma = stats.gamma.pdf(x_gamma, a=fit_alpha, scale=fit_beta, loc=fit_loc)
#print(fit_alpha)
#print(fit_loc)
#print(fit_beta)
#plt.plot(x_gamma, y_gamma, label='Gamma')
#print(s_plot)
#print(s_bin)
#print()
#print(y_gamma)
#print(x_gamma)
size = len(s_plot)

dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              #'norm',
              #'pearson3',
              #'triang',
              #'uniform',
              'weibull_min', 
              'weibull_max']

# Set up empty lists to stroe results
chi_square = []
p_values = []

# Set up 50 bins for chi-square test
# Observed data will be approximately evenly distrubuted aross all bins
percentile_bins = np.linspace(0,12,12)
percentile_cutoffs = np.percentile(s_plot, percentile_bins)
observed_frequency, bins = (np.histogram(s_plot, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions

for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(s_plot)
    if distribution == 'gamma' or distribution == 'weibull_min' or distribution == 'lognorm':
        print(distribution,end=' ')
        print(param)    
    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(s_plot, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)    
    
    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
    
    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    chi_square.append(ss)
        
# Collate results and sort by goodness of fit (best at top)

results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results.sort_values(['chi_square'], inplace=True)
    
# Report results

print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
print (results)

'''
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(s_hist)
print(fit_alpha)
print(fit_loc)
print(fit_beta)
x_gamma = np.linspace (0, 12, 120)
y_gamma = stats.gamma.pdf(x_gamma, a=fit_alpha, scale=fit_beta, loc=fit_loc)
print(x_gamma, y_gamma)
plt.plot(x_gamma, y_gamma, label='Gamma')
print(s_bin)
fig, ax = plt.subplots(1, 1)
ax.hist(s_bin, s_hist, density=True, histtype='stepfilled', alpha=0.2)
'''

plt.ylabel('Survival Function of Distance')
plt.xlabel('Distance')
plt.legend(loc='upper right')
plt.ylim(ymin=0, ymax=1.5)
plt.xlim(xmin=0)

# Save Plot
#plt.savefig('plots/' + dataset + '/' + 'new_d' + '_' + '_' + str(s_w) + '_' + str(s_g) + '_' + str(s_s) + '.png')

# Display Plot
plt.show()

