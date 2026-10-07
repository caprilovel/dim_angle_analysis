# Fit tan^2(theta_single) ~= C/n per model x dataset, where theta_single is
# the (unobservable) angle between an n-sample DiM estimate and the true
# population direction. We never observe theta_single directly -- only the
# split-half angle theta_split (the angle between two disjoint n-sample
# DiM estimates from the same pool), which is a proxy for it.
#
# Under a small-angle / high-dimensional-noise approximation, the two
# n-sample estimates deviate from the true direction by independent,
# roughly equal-magnitude angular errors theta_single, so their angle
# between EACH OTHER is about sqrt(2) times as large as each one's angle to
# the truth:
#     theta_split ~= sqrt(2) * theta_single
#  => tan(theta_single) ~= tan(theta_split / sqrt(2))
#  => tan^2(theta_single) ~= C/n   (standard 1/n error-propagation scaling)
#
# C is then an effective inverse-SNR for that (model, dataset): small C
# means the DiM direction is well-estimated from few samples; large C or a
# poor 1/n fit (low R^2, or a fitted exponent far from -1 in a log-log fit)
# signals heterogeneity -- e.g. a dataset where different subpopulations of
# pairs point in different directions, so averaging in more samples doesn't
# shrink the angle the way i.i.d. noise would predict.
#
# Usage:
#   python fit_dim_angle_c.py --csv results/dim_angle_refusal_llama3_8B_instruct.csv --tag "Refusal / llama3"

import numpy as np
import pandas as pd
import argparse
from scipy.optimize import curve_fit


def tan2_model(n, C):
    return C / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, nargs='+',
                         help='one or more split_half-mode CSVs (n,mean,... columns)')
    parser.add_argument('--tag', type=str, nargs='+', default=None,
                         help='labels for each csv (default: filename)')
    parser.add_argument('--n_min_fit', type=int, default=5,
                         help='exclude very small n from the fit (small-angle approx breaks down)')
    args = parser.parse_args()

    tags = args.tag if args.tag else [c for c in args.csv]
    assert len(tags) == len(args.csv)

    print(f"{'tag':45s} {'C (fit)':>10s} {'R^2':>8s} {'log-log slope':>14s}  note")
    print("-" * 100)

    for csv_path, tag in zip(args.csv, tags):
        df = pd.read_csv(csv_path)
        df = df[df['n'] >= args.n_min_fit]

        theta_split_rad = np.radians(df['mean'].values)
        theta_single_rad = theta_split_rad / np.sqrt(2)
        tan2_theta = np.tan(theta_single_rad) ** 2
        n = df['n'].values.astype(float)

        # fit tan^2(theta_single) = C / n
        popt, _ = curve_fit(tan2_model, n, tan2_theta, p0=[1.0])
        C = popt[0]
        pred = tan2_model(n, C)
        ss_res = np.sum((tan2_theta - pred) ** 2)
        ss_tot = np.sum((tan2_theta - tan2_theta.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # log-log slope as an independent check: tan^2(theta) ~ n^alpha,
        # alpha should be close to -1 if the 1/n model is a good fit
        log_n = np.log(n)
        log_tan2 = np.log(np.clip(tan2_theta, 1e-12, None))
        slope, intercept = np.polyfit(log_n, log_tan2, 1)

        note = ""
        if r2 < 0.9:
            note = "<- poor 1/n fit: likely heterogeneous pool (subgroup structure, not i.i.d. noise)"
        elif abs(slope + 1) > 0.15:
            note = f"<- slope {slope:.2f} far from -1: decay rate deviates from classic 1/n"

        print(f"{tag:45s} {C:10.4f} {r2:8.3f} {slope:14.3f}  {note}")


if __name__ == "__main__":
    main()
