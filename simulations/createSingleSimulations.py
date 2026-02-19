import os

import numpy as np
from scipy.stats import lognorm, norm
import pandas as pd

var = [(60261.15894485,
        60267.1960081,
        60280.01603645,
        60282.1209915,
        60285.08832445,
        60286.2156634,
        60288.14334374999,
        60290.04762965,
        60291.09341605,
        60483.36158255,
        60493.3765465,
        60515.3645086,
        60533.2917165,
        60557.16543785,
        60588.2037826,
        60593.20115745,
        60595.066361,
        60596.118592,
        60601.0592093,
        60602.10790415,
        60603.13498095,
        60862.33277365,
        60895.37816019999,
        60916.27517055,
        60935.26327725,
        61031.1563122),
       (60246.142597,
        60261.1811032,
        60267.0623839,
        60268.112685,
        60280.1223466,
        60280.14078775,
        60282.0989067,
        60285.1090485,
        60288.18959525,
        60290.06894145,
        60291.1144445,
        60486.4168379,
        60495.32219695,
        60518.2559711,
        60536.20490145,
        60557.31320916666,
        60588.2545364,
        60594.0614551,
        60595.1334355,
        60598.2826235,
        60601.1297719,
        60602.19812025,
        60603.2484175,
        60871.3534073,
        60898.23041555,
        60916.3225552,
        60936.10173374999,
        61055.05957925),
       (60246.12028475,
        60248.1702796,
        60256.1754323,
        60261.07239355,
        60267.03777155,
        60268.13490725,
        60280.16430990001,
        60286.19445685,
        60288.1648379,
        60483.40812325,
        60493.39732985,
        60517.2182734,
        60533.31248940001,
        60557.23455805,
        60588.2270131,
        60594.01299245,
        60595.08908615,
        60596.16809485,
        60601.0819518,
        60602.1547385,
        60603.18166735,
        60862.35481595001,
        60898.20829055,
        60916.2971828,
        60936.15266135,
        61032.188684),
       (60242.1491691,
        60247.2627569,
        60256.22288355,
        60261.02798815,
        60262.1121194,
        60267.1730938,
        60270.0532016,
        60280.03761045,
        60281.1177963,
        60520.33321335,
        60523.3402996,
        60525.39106720001,
        60527.3564361,
        60557.2123,
        60588.17215135,
        60596.0507189,
        60600.0777424,
        60602.06228965,
        60603.09196465,
        60606.14673035,
        60608.07532445,
        60862.40334505,
        60895.350411,
        60916.2495625,
        60935.1677282,
        61029.1687028),
       (60242.1242982,
        60247.24008975,
        60254.3470382,
        60256.2658885,
        60261.13559625,
        60267.1176993,
        60281.0522195,
        60285.17875025,
        60289.0717788,
        60290.11867115,
        60520.31076705,
        60523.36468175,
        60525.4116105,
        60527.37631475,
        60557.34286400001,
        60588.1486053,
        60594.2049036,
        60596.0719849,
        60600.04736975,
        60601.15232730001,
        60602.2193258,
        60603.22487555,
        60862.38216605,
        60895.32757135,
        60916.20206035,
        60935.13970545,
        61029.14020075),
       (60219.1047467,
        60220.25897425,
        60242.05382495,
        60245.0220162,
        60248.1489906,
        60252.2123311,
        60256.28887635,
        60261.05020405,
        60285.19908235,
        60483.34144425,
        60486.39609065,
        60495.3592128,
        60506.3338431,
        60557.25741775001,
        60588.07913785,
        60591.207025,
        60595.02159895,
        60596.0934419,
        60600.10388345,
        60602.0828421,
        60603.112756,
        60854.31636155,
        60895.26318265,
        60904.4044524,
        60935.0929223),
       (60242.0749789,
        60247.18535555,
        60256.19937265,
        60261.0933339,
        60267.1490223,
        60281.09754925,
        60285.15793735,
        60289.0487714,
        60290.09477365,
        60483.3823489,
        60493.35083185,
        60515.33935535001,
        60533.26803775,
        60557.18760835,
        60588.10253400001,
        60591.22995935,
        60595.0437298,
        60596.14199445,
        60600.1254241,
        60602.12953885,
        60603.1571405,
        60861.2952597,
        60895.28419424999,
        60916.2245404,
        60935.23726645,
        61029.08908835),
       (60242.1008552,
        60247.21295195,
        60256.2443856,
        60261.11421625,
        60262.13414895,
        60267.09079535,
        60270.03079415,
        60281.0757654,
        60285.1315991,
        60518.2774454,
        60520.2767181,
        60523.387478,
        60526.20815415,
        60557.278531,
        60588.12620695,
        60594.0347366,
        60595.11242705,
        60598.26084995,
        60601.107141,
        60602.17610695,
        60603.20442085,
        60862.31035535,
        60895.3057404,
        60916.17775965,
        60935.11558975,
        61029.1117592)]


def simulate_system_refined(model_rv_values, shape, loc, scale, intra_cv=0.25):
    """
    model_rv_values: array of noise-free RVs
    intra_cv: The target variation within the system (defaulting to your 24.89% result)
    """
    n_obs = len(model_rv_values)

    # 1. Sample ONE base sigma for the whole system from your Log-Normal fit
    # This ensures the "System Size" follows your DR5 distribution
    base_sigma = lognorm.rvs(s=shape, loc=loc, scale=scale, size=1)[0]

    # 2. Add "Flutter" per observation
    # We sample from a Normal distribution centered at 1.0 with a std of intra_cv
    flutter = np.random.normal(loc=1.0, scale=intra_cv, size=n_obs)

    # Ensure no negative sigmas (though unlikely with CV=0.25)
    flutter = np.maximum(flutter, 0.1)

    # 3. Calculate final point-by-point sigmas
    point_sigmas = base_sigma * flutter

    # 4. Generate the RV noise
    noise = np.random.normal(loc=0, scale=point_sigmas)
    simulated_rvs = model_rv_values + noise

    return simulated_rvs, point_sigmas

def simulate_system(model_rv_values, shape, loc, scale):
    """
    model_rv_values: array of noise-free RVs (e.g., from a Keplerian model)
    """
    n_obs = len(model_rv_values)

    # 1. Sample a unique noise amplitude for every single observation
    # This reflects the 25% CV you found in the real data
    point_sigmas = lognorm.rvs(s=shape, loc=loc, scale=scale, size=n_obs)

    # 2. Generate Gaussian noise scaled by each individual sigma
    noise = np.random.normal(loc=0, scale=point_sigmas)

    # 3. Add to the theoretical model
    simulated_rvs = model_rv_values + noise

    return simulated_rvs, point_sigmas


def sample_gamma(mean_gamma = 168, std_gamma = 35):
    gamma = norm.rvs(loc=mean_gamma, scale=std_gamma, size=1)
    return gamma

SIGMA_SHAPE = 0.8207
SIGMA_LOC =  0.0000e+00
SIGMA_SCALE = 1.9819

N = 5
output_path = "/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/CCF/simulations_light/"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

saved_systems = 0
digits_in_max_real = len(str(N*len(var)))
for i in var:
    # Drop NaNs in case some fields have fewer observations than others
    current_mjds = i
    tmp_rvs = np.zeros_like(current_mjds)

    for j in range(N):
        # 1. Sample errors and noise from your lognormal fit
        cur_rvs, cur_rv_sig = simulate_system_refined(tmp_rvs, SIGMA_SHAPE, SIGMA_LOC, SIGMA_SCALE)

        # 2. Add the systemic velocity (gamma)
        cur_rvs_w_gamma = cur_rvs + sample_gamma()

        # 3. Create DataFrame with specific column names
        df = pd.DataFrame({
            "Mean RV": cur_rvs_w_gamma,
            "Mean RVsig": cur_rv_sig,
            "MJD": current_mjds,
            'SNR_PPL': 100*np.ones_like(cur_rvs_w_gamma),
            'SNR': 100*np.ones_like(cur_rvs_w_gamma),
        })

        # 4. Save to CSV
        # Added 'i' to the filename to prevent systems from different fields
        # overwriting each other if you loop through multiple columns.
        filename = f"SIMuLaTioN_{saved_systems:0{digits_in_max_real}d}_CCF_RVs.csv"
        out = os.path.join(output_path, filename)
        df.to_csv(out, index=False)

        saved_systems += 1




