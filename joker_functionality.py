import thejoker as tj
# import astropy.table as at
import astropy.units as u
# import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
import pandas as pd
# from astropy.visualization.units import quantity_support
import make_RVs
import os
import sys
sys.maxsize = float('inf')

# os.environ["QT_QPA_PLATFORM"] = ""
os.environ["XDG_SESSION_TYPE"] = "xcb"

rvs_1, ts, errs_v1 = make_RVs.out_single_and_plot(0, 205, 0.6, 30, 30, 30, 20, 30, 3.0, False)

astropy_rvs = u.quantity.Quantity(rvs_1, unit=u.km / u.s)
astropy_err_rvs = u.quantity.Quantity(np.ones(30) * 3.0, unit=u.km / u.s)

# rnd = np.random.default_rng(seed=42)

t = Time(ts + 60347, format="mjd", scale="tcb")

# df = pd.DataFrame(np.stack([ts, rvs_1, errs_v1]).T, columns= [ 'ts', "rvs", 'errs_v1'])

# df.to_csv("/media/sf_Roey\'s/Masters/General/Scripts/scriptsOut/tomer_df.csv")
data = tj.RVData(t=t, rv=astropy_rvs, rv_err=astropy_err_rvs)
# _ = data.plot()
# plt.show()

prior = tj.JokerPrior.default(
    P_min=2 * u.day,
    P_max=1e3 * u.day,
    sigma_K0=30 * u.km / u.s,
    sigma_v=100 * u.km / u.s,
)

prior_samples = prior.sample(size=2500000, rng=rnd)


joker = tj.TheJoker(prior, rng=rnd)
joker_samples = joker.rejection_sample(data, prior_samples, max_posterior_samples=256)
print(joker_samples.tbl)
_ = tj.plot_rv_curves(joker_samples, data=data)
# plt.show()
