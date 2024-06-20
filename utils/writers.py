import os
import h5py
import numpy as np
from itertools import product


def write_charge_density_hdf5_qe_with_gt_mills(rhor, mill_gt, out_dir):

    #FFT to reciprocal space
    nr = rhor.shape
    n = np.prod(nr)
    rhog = np.fft.fftn(rhor) / n

    #Shift zero mode to center
    shift = [(n // 2) + 1 for n in nr]
    rhog = np.roll(rhog, shift, (0,1,2))
    na, nb, nc = nr

    rhog = rhog.flatten()
    #Create initial candidates of miller indices
    ijk_iter = np.array(list(
        product(
            range(-na//2, na//2), range(-nb//2, nb//2), range(-nc//2, nc//2)
        )
    ))
    ###  IMPORTANT: QE is quite particular about the order in which
    #    reciprocal charge density coefficients are written.
    #    Here we reorder the coeffs to match the ground truths.
    #    Ideally a better interface with QE would be found
    #    We leave this for future work.

    ##Find reordering of new data into ground-truth array
    #One-dimensionalize integer triples (i,j,k) as strings "i j k"
    def stringify(ijk):
        return " ".join([str(x) for x in ijk])
    mill_str1 = np.array([stringify(ijk) for ijk in ijk_iter])
    mill_str2 = np.array([stringify(ijk) for ijk in mill_gt])
    #Intersect candidate and GT set
    mask1 = np.isin(mill_str1, mill_str2)
    #Add missing elements to candidate set
    mask2 = np.isin(mill_str2, mill_str1, invert=True)
    missing_str = mill_str2[mask2]
    mill_str11 = np.concatenate([mill_str1[mask1], missing_str])
    missing_ijk = np.array(
        [[int(x) for x in s.split(" ")] for s in missing_str]
    )
    ijk_iter = np.concatenate([ijk_iter, missing_ijk])
    num_miss = len(missing_str)
    rhog = np.concatenate([rhog, np.zeros(num_miss)])
    mask11 = np.concatenate([mask1, np.ones(num_miss).astype('bool')])
    ##Find permutation which reorders candidates the same as ground-truth
    sort1 = mill_str11.argsort()
    sort2 = mill_str2.argsort().argsort()
    #Apply reduction and sort
    mill = ijk_iter[mask11][sort1][sort2]
    rhog_cut = rhog[mask11][sort1][sort2]

    ngm_g = len(rhog_cut)
    rhog_cut = np.array(rhog_cut).reshape(-1,1)

    ##Reformat complex-valued array as QE does
    rhog_cut = np.concatenate([rhog_cut.real, rhog_cut.imag], axis=1)
    rhog_cut = rhog_cut.reshape(-1)

    #Write to HDF5
    out_fn = 'charge-density.hdf5'
    out_fp = os.path.join(out_dir, out_fn)
    with h5py.File(out_fp, 'w') as hf5:
        hf5.create_dataset('MillerIndices', data=mill.astype('int32'))
        hf5.create_dataset('rhotot_g', data=rhog_cut.astype('<f8'))
        hf5.attrs.create('ngm_g', np.int32(ngm_g))
        hf5.attrs.create('nspin', np.int32(1))
        hf5.attrs.create('gamma_only', np.bytes_('.FALSE.'))

