import math
import numpy as np
from scipy import special

def calc_PDC_unbiased(f, A, vals, times, errors=[]):
    '''
    This function calculates the unbiased PDC value for a given frequency.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          f: float, a specific frequency value for it the PDC value is calculated
    '''

    z = np.array([times])
    mat_phase_delta = z.T - z
    mat_phase_delta = mat_phase_delta % (1 / f)

    b = mat_phase_delta * (1 / f - mat_phase_delta)

    mat_A_unbiased = A
    mat_B_unbiased = unbiased_u_centering(b, len(vals))

    # numerator part
    A = np.array(mat_A_unbiased)
    B = np.array(mat_B_unbiased)
    num = np.sum(np.multiply(A, B))

    # denominators part
    den_A = np.sum(A ** 2)
    den_B = np.sum(B ** 2)

    den = np.sqrt(den_A * den_B)

    res = num / den

    return res

def unbiased_u_centering(a, n):
    '''
    :param a: U-centered matrix (numpy nXn array)
    :param n: integer (length of a...)
    :return: U-center a distance matrix
    '''

    # Sum of rows
    v2_vec = np.array([np.sum(x) / (n - 2) for x in a])
    v2     = v2_vec.repeat(n).reshape((-1, n))

    # Sum of cols
    v3_vec = np.array([np.sum(x) / (n - 2) for x in a.T])
    v3 = v3_vec.repeat(n).reshape((-1, n)).T

    # Total sum
    tot_sum = np.sum(v3_vec)/(n-1)
    v4 = np.full((n, n), tot_sum)

    # Do the U-centering and make sure that the diagonal is zero.
    mat_unbiased = a - v2 - v3 + v4
    for i in np.arange(n):
        mat_unbiased[i,i] = 0

    return mat_unbiased


def calc_pdc_distance_matrix(vals, errors=[]):
    '''
    This function calculates the distance matrix used for the PDC calculation.
    Input:
          periodicity_detector: PeriodicityDetector class for which the PDC is calculated
          calc_biased_flag: bool, sets calculating the distance matrix for the biased PDC
          calc_unbiased_flag: bool, sets calculating the distance matrix for the unbiased PDC
    '''

    a = [[None for _ in range(len(vals))] for _ in
         range(len(vals))]

    for i, i_val in enumerate(vals):
        for j, j_val in enumerate(vals):
            if i > j:
                a[i][j] = a[j][i]
            elif i == j:
                a[i][j] = 0
            else:
                if len(errors)==0:
                    a[i][j] = abs(i_val - j_val)
                else:
                    y1 = i_val
                    y2 = j_val
                    e1 = errors[i]
                    e2 = errors[j]

                    den = np.sqrt(2 * (e1 ** 2 + e2 ** 2))

                    x = (y1 - y2) / den
                    y = (e1 + e2) / den

                    e = np.sqrt(e1 ** 2 + e2 ** 2) * (math.exp(-x ** 2) + x * special.erf(x) - y)

                    a[i][j] = np.sqrt(e)


    A = unbiased_u_centering(np.array(a), len(vals))

    return A, a

def calc_time_distance_matrix(times):
    n = len(times)
    a = np.zeros((n, n), dtype=float)
    t = np.asarray(times, float)
    for i in range(n):
        for j in range(i+1, n):
            d = abs(t[i] - t[j])
            a[i, j] = a[j, i] = d
    A = unbiased_u_centering(a, n)
    return A  # U-centered time-distance matrix

def pdc_window(times, pmin=1., pmax=1000., df=5e-4, samples_per_peak=10):
    # Frequency grid — linear in frequency, matching pdc_opt convention
    from period_search.periodogram import make_pdc_freq_grid
    T_baseline = float(np.ptp(times))
    freq = make_pdc_freq_grid(pmin, pmax, T_baseline, samples_per_peak)
    # A comes from TIME distances (not values!)
    A = calc_time_distance_matrix(times)

    # Compute PDC vs frequency using the same B definition you have
    out = np.empty_like(freq, dtype=float)
    z = np.array([times])
    for k, f in enumerate(freq):
        mat_phase_delta = (z.T - z) % (1.0/f)
        b = mat_phase_delta * (1.0/f - mat_phase_delta)
        B = unbiased_u_centering(b, len(times))

        num = np.sum(A * B)
        den = np.sqrt(np.sum(A**2) * np.sum(B**2))
        out[k] = num / den if den > 0 else np.nan

    return freq, out


def calc_pdc(freqs, times, vals, errors=[]):
    
    A, a = calc_pdc_distance_matrix(vals, errors)

    pdc_res_power_unbiased = freqs.copy()
    
    for index, f in enumerate(freqs):
        pdc_res_power_unbiased[index] = calc_PDC_unbiased(f, A, vals, times)

    return A, a, pdc_res_power_unbiased


import math
import numpy as np
from scipy import special
from numba import njit, prange


# --- Optimized Helper Functions ---

def unbiased_u_centering_fast(a):
    """
    Vectorized version of unbiased_u_centering.
    Removes Python loops and unnecessary list comprehensions.
    """
    n = a.shape[0]

    # Axis 1 = sum over columns (for each row)
    # Axis 0 = sum over rows (for each column)
    sum_rows = np.sum(a, axis=1)
    sum_cols = np.sum(a, axis=0)
    total_sum = np.sum(sum_rows)

    # Broadcast subtraction: A - RowMean - ColMean + GrandMean
    # Note: The original math divided by (n-2) for rows/cols and (n-1)(n-2) for total
    v2 = sum_rows.reshape(-1, 1) / (n - 2)
    v3 = sum_cols.reshape(1, -1) / (n - 2)
    v4 = total_sum / ((n - 1) * (n - 2))

    mat_unbiased = a - v2 - v3 + v4

    # Set diagonal to 0
    np.fill_diagonal(mat_unbiased, 0.0)

    return mat_unbiased


def calc_pdc_distance_matrix_fast(vals, errors=None):
    """
    Fully vectorized distance matrix calculation.
    """
    vals = np.array(vals)
    n = len(vals)

    if errors is None or len(errors) == 0:
        # Broadcasting to create difference matrix |val_i - val_j|
        #
        a = np.abs(vals[:, None] - vals[None, :])
    else:
        errors = np.array(errors)
        y1 = vals[:, None]  # Column vector
        y2 = vals[None, :]  # Row vector
        e1 = errors[:, None]
        e2 = errors[None, :]

        # Calculate den: sqrt(2 * (e1^2 + e2^2))
        # We perform the squaring and addition once via broadcasting
        e_sq_sum = e1 ** 2 + e2 ** 2
        den = np.sqrt(2 * e_sq_sum)

        # Avoid division by zero on diagonal or if errors are 0 (though unlikely in valid data)
        # We can use a mask or added epsilon, but sticking to logic:
        # If den is 0, x is undefined. Assuming errors > 0.

        x = (y1 - y2) / den
        y = (e1 + e2) / den

        # Vectorized Error Function
        # Note: standard math.exp/erf doesn't work on arrays, use scipy or numpy equivalents
        # However, since we want to JIT this later, we might separate this logic.
        # For now, using scipy.special.erf is fast enough for the setup phase.

        term1 = np.exp(-x ** 2)
        term2 = x * special.erf(x)

        e = np.sqrt(e_sq_sum) * (term1 + term2 - y)
        a = np.sqrt(np.maximum(e, 0))  # Ensure non-negative before sqrt

    # Ensure symmetry and zero diagonal (numerical precision might drift)
    np.fill_diagonal(a, 0.0)

    # U-centering
    A = unbiased_u_centering_fast(a)
    return A, a


# --- Numba Optimized Kernel ---

@njit(parallel=True, fastmath=True)
def calculate_pdc_loop(freqs, time_diff_matrix, A):
    """
    This function replaces the loop inside calc_pdc.
    It runs in parallel across all CPU cores.
    """
    n = len(A)
    n_freqs = len(freqs)
    pdc_res = np.empty(n_freqs, dtype=np.float64)

    # Pre-calculate constants for U-centering inside the loop
    # We can't use the python function inside Numba easily without overhead,
    # so we inline the math for maximum speed.
    div_n2 = 1.0 / (n - 2)
    div_n1n2 = 1.0 / ((n - 1) * (n - 2))

    for k in prange(n_freqs):
        f = freqs[k]
        inv_f = 1.0 / f

        # 1. Calculate B matrix elements (phase distance)
        # mat_phase_delta = mat_phase_delta % (1 / f)
        # We use the precomputed time differences
        mat_phase_delta = np.remainder(time_diff_matrix, inv_f)

        # b = mat_phase_delta * (1 / f - mat_phase_delta)
        b = mat_phase_delta * (inv_f - mat_phase_delta)

        # 2. Unbiased U-centering of B (Inlined for speed)
        # Sums
        row_sums = np.sum(b, axis=1)
        # col_sums are same as row_sums for symmetric matrix,
        # but phase diff modulo might break perfect symmetry?
        # No, |ti-tj| is symmetric.
        # But (ti-tj)%p is NOT symmetric! (5-2)%3 = 0, (2-5)%3 = -3%3 = 0.
        # Wait, (z.T - z) produces antisymmetry.
        # The original code: mat_phase_delta = (z.T - z) % (1/f).
        # Modulo of negative numbers in Python: -1 % 5 = 4.
        # So it is NOT symmetric. We must compute col sums separately.
        col_sums = np.sum(b, axis=0)
        total_sum = np.sum(row_sums)

        # We need to construct B_unbiased to get the denominator term
        # Numerator term sum(A * B_unbiased)

        # Optimization: calculating Frobenius product
        num = 0.0
        den_B = 0.0

        for i in range(n):
            v2 = row_sums[i] * div_n2
            for j in range(n):
                if i == j:
                    val_b = 0.0  # Diagonal forced to 0
                else:
                    v3 = col_sums[j] * div_n2
                    v4 = total_sum * div_n1n2
                    val_b = b[i, j] - v2 - v3 + v4

                # Accumulate Numerator
                num += A[i, j] * val_b

                # Accumulate Denominator part B
                den_B += val_b * val_b

        # 3. Final Calculation
        den_A = np.sum(A ** 2)  # This is constant, could be passed in, but it's fast
        den = np.sqrt(den_A * den_B)

        if den == 0:
            pdc_res[k] = 0.0
        else:
            pdc_res[k] = num / den

    return pdc_res


def calc_pdc_optimized(freqs, times, vals, errors=[]):
    # 1. Setup Distance Matrix (Vectorized, runs on CPU once)
    A, a = calc_pdc_distance_matrix_fast(vals, errors)

    # 2. Pre-calculate Time Difference Matrix
    # This prevents O(N^2) allocations inside the loop
    times_arr = np.array(times)
    time_diff_matrix = times_arr[:, None] - times_arr[None, :]

    # 3. Run Parallel Loop (Compiled)
    pdc_res_power_unbiased = calculate_pdc_loop(freqs, time_diff_matrix, A)

    return A, a, pdc_res_power_unbiased