import sympy as sp
import pandas as pd
import re
import os
from pathlib import Path

# calculates the min mass of the companion from the measured P, e, K1 and the mass of the star from the mass_bloem.csv file
mass_file = '/Users/roeyovadia/Downloads/mass_bloem.csv'
root = Path('/Users/roeyovadia/Roey/Masters/Reasearch/scriptsOut/OrbitalFitting/bla/')

def param(lines, key, cast=float):
    pat = rf'^\s*{re.escape(key)}\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)'
    m = lines.str.extract(pat, expand=False).dropna()
    if m.empty:
        raise KeyError(f"'{key}' not found in report")
    return cast(m.iloc[0])


#def read_mass(mass_file):
    #masdf = pd.read_csv(mass_file, header=0)
    #return print(masdf['Mspec'],)

results= []
for sub in root.iterdir():
    if not sub.is_dir():
        continue
    report = sub / f'{sub.name}_report.txt'
    if not report.exists():
        print(f"Report not found for {sub.name}")
        continue

    # lines = pd.read_csv(report, header=None).iloc[:, 0]
    P = 92.59259
    k1 = 53.6451216
    ecc = 0.44855506
    f = 1.036149e-7
    masdf = pd.read_csv(mass_file, header=0)
    try :
        mspec = masdf.loc[masdf['ID'] == f'BLOeM_{sub.name}' , 'Mspec'].iloc[0]
        print(f"Mspec({sub.name}) = {mspec:.5f}")
    except IndexError:
        print(f"{sub.name} not found in mass_bloem.csv")

    mspec_err_plus = masdf.loc[masdf['ID'] == f'BLOeM_{sub.name}', 'Mspec_er_plus'].iloc[0]
    print(f"Mspec_er_plus({sub.name}) = {mspec_err_plus:.5f}")
    mspec_err_min = masdf.loc[masdf['ID'] == f'BLOeM_{sub.name}', 'Mspec_er_minus'].iloc[0]
    f = (P * k1 ** 3 * (1 - ecc ** 2) ** (3 / 2)) * 1.036149e-7
    m2 = sp.symbols('M2', positive=True)
    eq = f * (mspec + m2) ** 2 - (m2 ** 3)
    eq2 = f * ((mspec_err_plus+mspec) - m2) ** 2 - (m2 ** 3)
    eq3 = f * ((mspec+mspec_err_min)- m2) ** 2 - (m2 ** 3)
    M2 = sp.nsolve(eq, 34)
    M2_plus = sp.nsolve(eq2, 100)
    M2_minus = sp.nsolve(eq3, 15)
    print(f"M2 min({sub.name}) = {M2:.5f}")
    print(f"M2 min upper limit({sub.name}) = {M2_plus:.5f}")
    print(f"M2 min lower limit({sub.name}) = {M2_minus:.5f}")

    results.append({
        "system": sub.name,
        "M2_Msun": float(M2),
        "M2_plus_Msun": float(M2_plus),
        "M2_minus_Msun": float(M2_minus)
    })


pd.DataFrame(results, columns=['system', 'M2_Msun','M2_plus_Msun','M2_minus_Msun']).to_csv('binary_masses.csv', index=False)