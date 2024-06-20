import os
import h5py
import json
import numpy as np
from itertools import product
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.espresso import read_espresso_in, read_espresso_out
from pyrho.charge_density import ChargeDensity
from pyrho.pgrid import PGrid


INIT_STRUCTURE_DIR = "metadata/init-structures"
DENSITY_DIR = "data/aiida"


def lookup_workdir(mp_id, metadata_fp):
    found = False
    with open(metadata_fp, 'r') as fh:
        for line in fh.readlines():
            mp_id_, uuid = line.strip().split(" ")
            if mp_id == mp_id_:
                found = True
                break
    assert found, "No matches found!"
    work_dir = os.path.join(DENSITY_DIR, uuid[0:2], uuid[2:4], uuid[4:])
    return work_dir


def load_structure(mp_id):
    """
    Load MP structure
    """
    fn = '{}.json'.format(mp_id)
    fp = os.path.join(INIT_STRUCTURE_DIR, fn)
    if os.path.exists(fp):
        struct = Structure.from_file(fp)
        return struct
    else:
        print("Structure for {} not found".format(mp_id))
        return None


def read_charge_density_hdf5_qe(data_fp, nr=None):
    """
    Gratefully modified from:
    * https://github.com/QEF/postqe/blob/master/postqe/charge.py
    """

    with h5py.File(data_fp, "r") as h5f:
        MI = np.array(h5f.get('MillerIndices'))
        ngm_g = h5f.attrs.get('ngm_g')
        aux = np.array(h5f['rhotot_g']).reshape([ngm_g, 2])
        rhotot_g = aux.dot([1.e0,1.e0j])

    if not nr:
        nr = [2*max(abs(MI[:,i]))+1 for i in range(3)]

    rho_temp = np.zeros(nr, dtype=np.complex128)
    for (i,j,k), rho in zip( MI,rhotot_g):
        rho_temp[i,j,k] = rho

    rhotot_r = np.fft.ifftn(rho_temp) * np.prod(nr)
    return rhotot_r.real, MI, rhotot_g


def load_charge_density_qe(mp_id, work_dir, nr=None):

    #Load structure info from QE
    fn = 'aiida.in'
    fp = os.path.join(work_dir, fn)
    atoms = read_espresso_in(fp)

    #build path to density file. check it exists
    data_dir = os.path.join(work_dir, "out/aiida.save/")
    data_fn = "charge-density.hdf5"
    data_fp = os.path.join(data_dir, data_fn)
    assert os.path.exists(data_fp), "Charge density file not found!"

    ##Read QE charge-density
    rho, MI, rhotot_g = read_charge_density_hdf5_qe(data_fp, nr=nr)
    grid_shape = rho.shape

    #Unpack needed data
    formula = atoms.symbols
    lat_mat = atoms.cell
    species = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    struct = Structure(lat_mat, species, pos)

    #Load density into pyrho
    pgrid_tot  = PGrid(rho, lat_mat)
    #NB: magnetization not implemented
    rho_diff = np.zeros(rho.shape)
    pgrid_diff = PGrid(rho_diff, lat_mat)
    pgrids = { 'total': pgrid_tot, 'diff': pgrid_diff }
    Rho = ChargeDensity(pgrids, struct, normalization=None)
    rho = Rho.normalized_data['total']

    #Build coordinate grid
    lat_mat = Rho.lattice
    vecs = [ np.linspace(0, 1, grid_shape[i], endpoint=False) for i in range(3) ]
    gridded = np.meshgrid(*vecs, indexing="ij")
    res = np.dot(lat_mat.T, [g_.flatten() for g_ in gridded])

    data = {
             'x': res[0,:],
             'y': res[1,:],
             'z': res[2,:],
             'charge-density': rho,
             'atoms': atoms,
             'structure': struct,
             'pyrho': Rho,
             'millers': MI,
             'charge-density-fourier': rhotot_g,
             'grid-shape': grid_shape
    }

    return data


def load_charge_density(data_dir, mp_id, which='qe', metadata_fp=None, nr=None):
    assert metadata_fp, "Must specify metadata file  for QE!"
    work_dir = lookup_workdir(mp_id, metadata_fp)
    data = load_charge_density_qe(mp_id, work_dir, nr)
    return data


def load_predicted_density(results_dir, grid_shape):
    data_fn = "is2re_predictions.npz"
    data_fp = os.path.join(results_dir, data_fn)

    with np.load(data_fp) as f:
        rho_pred = f['energy']
        pred_ids = f['ids']

    #NB: sort must sort by prediction ID
    _ = sorted(zip(pred_ids, rho_pred), key=lambda x: int(x[0].split("_")[-1]))
    rho_pred = np.array([x[1] for x in _])
    rho_pred = rho_pred.reshape(grid_shape)
    return rho_pred
