import os
import sys
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

import sys; sys.path.append("/home/pepope/work/rho-learn/utils/")
from loaders import load_charge_density
from preprocessing import (
  preprocess_one,
  resample_high_density
)


@registry.register_dataset("hdf5")
class Hdf5Dataset(Dataset):
    r"""Dataset class to load from hdf5 files containing
    charge densities.

    Useful for the Structure to Density (S2D) task.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(Hdf5Dataset, self).__init__()
        self.config = config
        #data preprocessing params
        self.p_high = self.config['p_high']
        self.p_unif = self.config['p_unif']
        self.init_ids()
        #optional resizing of grid at load time
        if config.get('grid_size'):
            self.nr = [int(x) for x in config['grid_size'].split(",")]
        else:
            self.nr = None
        num_structs = len(self.mp_ids)
        if num_structs > 0:
            print("Loading {} structures into memory...".format(num_structs))
            self.init_data()
            print("Data loading complete.")
        else:
            self.num_samples = 0
        #graph preprocessing params
        self.max_neigh = self.config['max_neigh']
        self.radius = self.config['radius']
        self.num_elements = self.config['num_elements']
        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neigh,
            radius=self.radius,
        )
        self.transform = transform

    def init_ids(self):
        """
        Init list of MP-IDs to load. Optionally accept list as
          * comma-seperated string (possibly only one)
          * textfile containing IDs (in first column) (must end in .txt)

        IDs will be diff'd against metadata to only return completed runs
        """
        #Load metadata for completed IDs
        self.aiida_metadata_file = self.config["metadata_file"]
        with open(self.aiida_metadata_file, 'r') as fh:
            completed_mp_ids = [x.split(" ")[0] for x in fh.readlines()]

        #Get list of IDs to load
        #check if input string is file or comma-seperated list
        mp_ids_str = self.config['mp_ids']
        if mp_ids_str[-4:] == ".txt":
            with open(mp_ids_str, 'r') as fh:
                mp_ids = [x.split(" ")[0] for x in fh.readlines()]
        else:
            mp_ids = [x.strip() for x in mp_ids_str.split(",") if x]

        #Check all IDs to load were completed
        mp_ids = [x for x in mp_ids if x in completed_mp_ids]
        self.mp_ids = mp_ids

    def init_data(self):
        """
        Warning: Loads all structure density data into memory for fast access.
                 Will OOM if there are too many structures.
        """
        atoms = []
        gridsizes = []
        for i,mp_id in enumerate(self.mp_ids):
            #load data
            data = load_charge_density(
              data_dir=None, mp_id=mp_id, metadata_fp=self.aiida_metadata_file, nr=self.nr
            )
            #get (xyz, rho) pairs
            xyz_ = np.concatenate([data[w].flatten()[:, np.newaxis]
                for w in ['x','y','z']], axis=1)
            rho_ = data['charge-density'].flatten()[:, np.newaxis]
            #resample. take all points if p_high=0 (no cutoff)
            if self.p_high > 0:
                raise NotImplementedError
                xyz_, rho_ = resample_high_density(
                    xyz_,
                    rho_,
                    p_high=self.p_high,
                    p_unif=self.p_unif
                )
            #add to collection
            if i == 0:
                xyz = xyz_
                rho = rho_
            else:
                xyz = np.concatenate([xyz, xyz_], axis=0)
                rho = np.concatenate([rho, rho_], axis=0)
            #track atoms
            atoms_ = data['atoms']
            atoms.append(atoms_)
            #track gridsizes for lookup
            N = rho_.shape[0]
            gridsizes.append(N)

        self.rho = rho
        self.xyz = xyz
        self.atoms = atoms
        self.gridsizes = gridsizes
        length = self.rho.shape[0]
        self._keys = [
            f"{j}".encode("ascii")
            for j in range(length)
        ]
        self.num_samples = len(self._keys)

    def lookup_structure_index(self, idx):
        """
        Find the index to which structure a global index corresponds
        """
        for sidx,N in enumerate(self.gridsizes):
           if idx < N:
              break
        return sidx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sidx = self.lookup_structure_index(idx)
        atoms = self.atoms[sidx]
        mp_id = self.mp_ids[sidx]
        graph = preprocess_one(
            idx, mp_id, self.xyz[idx], self.rho[idx],
            atoms, self.a2g, self.num_elements
        )
        data_object = pyg2_data_transform(graph)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object
