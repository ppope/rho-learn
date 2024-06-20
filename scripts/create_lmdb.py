"""
Script for creating LMDB datasets for pointwise charge density prediction

Modified from:
* https://github.com/Open-Catalyst-Project/ocp/blob/main/scripts/preprocess_relaxed.py
"""
import os
import copy
import lmdb
import random
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import multiprocessing as mp
import torch

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

from utils.loaders import load_charge_density
from utils.preprocessing import (
  preprocess_one,
  resample_near_atom
)


def write_one_density_chunk(args):
    (
        pid, data_id, start_idx,
        db_path, xyz, rho, atoms,
        no_tqdm, max_atomic_num,
        max_neigh, radius,
    ) = args

    N = rho.shape[0]
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    a2g = AtomsToGraphs(
        max_neigh=max_neigh,
        radius=radius,
    )

    if not no_tqdm:
       pbar = tqdm(
           total=N,
           position=pid,
           desc="Writing chunk of graph, density pairs to LMDB",
       )

    idx = start_idx
    #Add density points as records to DB
    for xyz_, rho_ in zip(xyz, rho):
        #Create record
        g = preprocess_one(
            idx, data_id, xyz_, rho_, atoms, a2g,  max_atomic_num
        )
        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(g, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1
        if not no_tqdm: pbar.update(1)
    db.close()


def main(args):

    if args.use_float64:
        torch.set_default_dtype(torch.float64)
    #make runs determinstic
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    out_dir = args.out_dir
    num_workers = args.num_workers
    mp_id = args.mp_id
    print(mp_id)
    no_tqdm = args.no_tqdm
    metadata_fp = args.metadata_fp
    num_unif = args.num_unif
    max_neigh = args.max_neigh
    radius = args.max_neigh
    max_atomic_num = args.max_atomic_num
    use_all = args.use_all
    preprocess_radius = args.preprocess_radius
    grid_size = args.grid_size
    if grid_size:
        nr = [int(x) for x in grid_size.split(",")]
        assert len(nr) == 3, "3D required!"
    else:
        nr = None

    assert metadata_fp, "Must specify metadata file for QE"

    # Init output dir, data ID, and output filename
    os.makedirs(out_dir, exist_ok=True)
    data_id = mp_id
    db_fn = "{}.lmdb".format(data_id)
    db_path = os.path.join(out_dir, db_fn)

    #Exit if exists
    if os.path.exists(db_path):
        print("LMDB file already exists! Exiting... \n{}".format(db_path))
        return

    #Load data
    data = load_charge_density(
        None, data_id,
        which='qe',
        metadata_fp=metadata_fp,
        nr=nr
    )
    atoms = data['atoms']
    struct = data['structure']

    #Flatten grids for iteration
    xyz = np.concatenate([data[w].flatten()[:, np.newaxis]
        for w in ['x','y','z']], axis=1)
    rho = data['charge-density'].flatten()[:, np.newaxis]

    n_orig = rho.shape[0]
    if not use_all:
        #Resample to emphasize high density points around atoms
        xyz, rho = resample_near_atom(
            xyz, rho, atoms, struct, num_unif, c=preprocess_radius
        )
        n_new = rho.shape[0]
        print("Resampled data from n={} to n={}".format(n_orig, n_new))
    else:
        print("Using all data with n={}".format(n_orig))

    #Chunk per worker
    xyz_chunks = np.array_split(xyz, num_workers)
    rho_chunks = np.array_split(rho, num_workers)

    #Pre-compute start indices for LMDB entry IDs
    #which must be unique per entry
    start_idxs = [0]
    for i in range(num_workers-1):
        x = start_idxs[i]
        n = rho_chunks[i].shape[0]
        start_idxs.append(x+n)

    #Define args for workers
    chunks = zip(start_idxs, xyz_chunks, rho_chunks)
    common_args = [atoms, no_tqdm, max_atomic_num, max_neigh, radius]
    mp_args = [
        [i, data_id, sidx, db_path, xyzc, rhoc] + common_args
            for i, (sidx, xyzc, rhoc) in enumerate(chunks)
    ]

    if num_workers > 1:
        #Parallelize writes
        pool = mp.Pool(num_workers)
        list(pool.imap(write_one_density_chunk, mp_args))
        pool.close()
    else:
        #Useful for debugging
        write_one_density_chunk(mp_args[0])


    ##Finalize DB
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    # Save count + stats of objects in lmdb.
    num_writes = rho.shape[0]
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"),
        pickle.dumps(num_writes, protocol=-1)
    )
    txn.commit()
    db.sync()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp_id', type=str,
        help='(Single) Materials Project ID to load')
    parser.add_argument('--out_dir', type=str,
        help='Output directory for LMDB files')
    parser.add_argument('--num_workers', type=int, default=1,
        help='Number of worker threads')
    parser.add_argument('--metadata_fp', type=str, default='',
        help='Metadata filepath for calculation work dir lookup')
    parser.add_argument('--no-tqdm', action='store_true',
        help="Don't use tqdm. Useful for sbatch scripts")
    parser.add_argument('--use-float64', action='store_true', default=False,
        help="Write data as float64s")
    parser.add_argument('--num_unif', type=int, default=0,
        help="Uniformly sample this many points")
    parser.add_argument('--max_neigh', type=int, default=50,
        help="Max number of neighbors for nodes")
    parser.add_argument('--radius', type=float, default=6.,
        help="Radius around nodes to decide edges (Angstrom)")
    parser.add_argument('--max_atomic_num', type=int, default=83,
        help="Max atomic number to consider")
    parser.add_argument('--use_all', action='store_true', default=False,
        help="Skip resampling and use all data")
    parser.add_argument('--preprocess_radius', default=1.5, type=float,
        help="When proprocessing, multipler on radius used to extract near-atom points")
    parser.add_argument('--grid_size', default=None, type=str,
        help="Comma-delimited string of integers specifying grid size. Warning: too small will cause problems")
    parser.add_argument('--num_total', type=int, default=0, help='ignored')
    args = parser.parse_args()
    main(args)
