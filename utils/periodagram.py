# from Scripts.make_RVs import out_single_and_plot
# from astropy.io import ascii
import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pylab as pylab
from Scripts.PDC.pdc_func import calc_pdc
from scipy.stats.distributions import chi2

params = {'legend.fontsize': 'large',
          'figure.figsize': (3, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

pmin = 1.
pmax = 2000.


def pdc(time, data, data_err=[], pmin=1., pmax=1000):
    freq = np.arange(1/pmax, 1/pmin, 0.0001)
    A, a, pdc_power_reg = calc_pdc(freq, time, data, data_err)
    fap1 = chi2.sf(np.max(pdc_power_reg) * len(time) + 1, 1) ## todo what???
    max_freq1 = freq[np.argmax(pdc_power_reg)]
    best_period1 = 1/max_freq1
    return best_period1, fap1, freq, pdc_power_reg

def ls(time, data, data_err=[], probabilities = [0.5, 0.01, 0.001], pmin=1., pmax=1000, norm='model',
       ls_method='fast', fa_method = 'baluev', samples_per_peak=1000, nterms=1, center_data = True):
    '''
    Wrapper around astropy's Lomb-Scargle. See:
    https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle
    for further options.
​
    Parameters:
            time: Times of observations
            data: Radial velocity or magnitudes/fluxes
            data_err: Error on data
            probabilities: The false alarm probabilities
            pmin: Minimum orbital period of the grid (grid is computed is frequency)
            pmax: Maximum orbital period of the grid
            norm: Normalization to use for the periodogram
            ls_method: The lomb scargle implementation to use
            fa_method: The approximation method to use for false alarm probability and false alarm level computation
            samples_per_peak: The approximate number of desired samples across the typical peak
​
    Returns:
            best_period1: orbital period corresponding to the stronger peak in the periodogram
            fap1: The false alarm probability of the highest peak
            fal1: The false alarm levels computed at the given probabilities
            frequency1: The list of frequencies of the periodogram
            power1: The list of powers of the periodogram
​
    '''
    from astropy.timeseries import LombScargle
    if data_err is not []:
        ls1 = LombScargle(time, data, data_err, normalization=norm, nterms=nterms, center_data=center_data)
    else:
        ls1 = LombScargle(time, data, normalization=norm , nterms=nterms, center_data=center_data)
    if nterms>1:
        fal1 = [0]*len(probabilities)
    else:
        fal1 = ls1.false_alarm_level(probabilities, method=fa_method)
    frequency1, power1 = ls1.autopower(method=ls_method, minimum_frequency=1/pmax,
                                                    maximum_frequency=1/pmin,
                                                    samples_per_peak=samples_per_peak)
    if nterms>1:
        fap1 = 0.0
    else:
        fap1 = ls1.false_alarm_probability(power1.max(), method=fa_method)
    # print('   FAP of the highest peak x100    : ', f'{fap1*100:.5f}')
    max_freq1 = frequency1[np.argmax(power1)]
    best_period1 = 1/max_freq1
    print("   Primary best Period             :  {0:.4f} days".format(best_period1))
    return best_period1, fap1, fal1, frequency1, power1



def plotls(frequency, power, fal, bins = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], star_id='Sim', pmin=1., pmax=1000.):
    '''
    Function to create publication-ready plot of the periodogram obtained with "ls"
​
    Parameters:
            frequency: The list of frequencies of the periodogram returned by "ls"
            power: The list of powers of the periodogram returned by "ls"
            fal: The false alarm levels returned by "ls"
            bins: list of ticks and tick labels for the orbital period axis
            star_id: Star name or identifyer
​
    Returns:
            Periodogram computed by "ls"
    '''

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(1/frequency, power, 'k-', alpha=0.5)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.set(xlim=(0.3, 2), ylim=(-0.03*power.max(), power.max()+0.1*power.max()),
        xlabel='Period (days)',
        ylabel='Lomb-Scargle Power')
    fig.suptitle("Primary best Period : {0:.3f} days".format(1/frequency[np.argmax(power)]))
    plt.xscale('log')
    tickLabels = map(str, bins)
    ax.set_xticks(bins)
    ax.set_xticklabels(tickLabels)
    print(fal)
    ax.plot( (0.5, 800), (fal[0], fal[0]), '-r', lw=1.2)
    ax.plot( (0.5, 800), (fal[1], fal[1]), '--y', lw=1.2)
    ax.plot( (0.5, 800), (fal[2], fal[2]), ':g', lw=1.2)
    ax.set_xlim(pmin, pmax)
    if power.max()+0.1*power.max() >= 10:
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))
    else:
        ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.1f}'))
    if star_id:
        #plt.title(' periodogram')
        plt.savefig(star_id+'_periodogram.png')
    plt.show()
    plt.close()


# if __name__ == '__main__':
#
#     v1s, hjds1, errs_v1 = out_single_and_plot(t0=0 ,p = 50, ecc = 0.3, omega = 10, k1=0, k2= 0, gamma=10, nrv=25, sig_rv=3.0, plot=True)
#     ervv1s = [3.] * 25
#
#     period, fap, fal, freq, pow = ls(hjds1, v1s,data_err=errs_v1, pmin=pmin, pmax=pmax)
#     plotls(freq, pow, fal, pmin=pmin, pmax=pmax)

probs = np.arange(-1, 1, 0.0001)
fap1 = chi2.sf(probs * 25 + 1, 1)  # Survival function of chi-square distribution

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(probs, fap1, color='blue', alpha=0.6, edgecolors='black', label='FAP1')

# Add labels and title
plt.xlabel('Probability (probs)', fontsize=14)
plt.ylabel('FAP1', fontsize=14)
plt.title('Scatter Plot of FAP1 vs. Probability', fontsize=16)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()