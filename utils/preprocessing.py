import copy
import numpy as np
from ase import Atom, data
from ase.geometry.geometry import get_distances


def resample_uniform(xyz, rho, size):
    """
    Randomly sample scalar field over volume.
    """
    num_points = rho.shape[0]
    if num_points < size:
        print(
          "Could not resample requested number of points {}".format(size)
        )
        print("Uniformly sampling {} instead".format(num_points))
        size = num_points

    all_inds = np.arange(num_points)
    rand_inds = np.random.choice(all_inds, size=size, replace=False)
    return xyz[rand_inds], rho[rand_inds]


def resample_high_density(xyz, rho, num_high, num_unif):
    """
    Resample to emphasize high-density points
        (1) Take all points with density value greater value at given percentile
        (2) Uniformly sample everything else
    """
    num_orig = xyz.shape[0]
    if num_orig < num_high:
        print(
          "Could not select requested number of high density points {}".format(num_orig)
        )
        print("Selecting {} instead".format(num_orig))
        num_high = num_orig

    #Take topk point with highest magnitude of density
    ##NB: PAW method sometimes yields negative densities. Include these.
    abs_rho = np.abs(rho).squeeze()
    high_density_inds = np.argsort(abs_rho)[-num_high:]
    not_high_density_inds = np.array(
        [i for i in range(num_orig) if not i in high_density_inds]
    )

    xyz_high_density = xyz[high_density_inds]
    rho_high_density = rho[high_density_inds]

    #Uniformly sample everything else
    xyz_unif, rho_unif = resample_uniform(
        xyz[not_high_density_inds], rho[not_high_density_inds],
        size=num_unif
    )

    #Concat and return
    xyz_new = np.concatenate([xyz_high_density, xyz_unif], axis=0)
    rho_new = np.concatenate([rho_high_density, rho_unif], axis=0)
    return xyz_new, rho_new


def resample_near_atom(xyz, rho, atoms, struct, num_unif, return_inds=False, c=1.5):
    """
    Extract points around atoms
        (1) Take all points within ionic radii each atom
        (2) Uniformly sample `num_unif` samples of everything else
    """
    cell = atoms.cell.array
    ##Collect atomic positions and radii
    positions = atoms.get_positions()
    atom_nums = atoms.get_atomic_numbers()
    ##Guess oxidation states per site to determine ionic radii
    struct_ = struct.copy()
    struct_.add_oxidation_state_by_guess()

    ##Determine which points are within radii of any atoms
    is_near_atom = np.zeros(xyz.shape[0], dtype=int)
    for pos, atom_num in zip(positions, atom_nums):

        ##To get ionic radius, first find matching site
        match_found = False
        for site in struct_.sites:
            _,dist2a = get_distances(
                pos, site.frac_coords, cell=cell, pbc=True
            )
            if dist2a == 0:
                match_found = True
                break
        if not match_found:
            raise Exception("No matching sites found!")
        rad = site.specie.ionic_radius

        #Use atomic radius when ionic is not found, (e.g. for H+)
        if not rad:
            symbol = site.specie.symbol
            print("Defaulting to atomic radius for {}".format(symbol))
            rad = site.specie.element.atomic_radius

        #Compute distances to atom and select nearby
        _, dist2a = get_distances(xyz, pos, cell=cell, pbc=True)
        is_near_atom_ = (dist2a <= c*rad).astype(int)
        is_near_atom += is_near_atom_.squeeze()

    near_atom_inds = is_near_atom > 0

    if return_inds:
        return near_atom_inds

    xyz_near_atom = xyz[near_atom_inds]
    rho_near_atom = rho[near_atom_inds]

    #Uniformly sample everything else
    xyz_unif, rho_unif = resample_uniform(
        xyz[~near_atom_inds], rho[~near_atom_inds], size=num_unif
    )

    #Concat and return
    xyz_new = np.concatenate([xyz_near_atom, xyz_unif], axis=0)
    rho_new = np.concatenate([rho_near_atom, rho_unif], axis=0)
    return xyz_new, rho_new


def preprocess_one(idx, mp_id, xyz, rho, atoms, a2g, max_atomic_num):
    """Preprocess a single density point from a given structure into a
       (graph, density) pair
    """
    #NB: We must copy atoms because we mutate it with the query point (QP).
    atoms_ = copy.deepcopy(atoms)
    #NB: Use a dummy symbol for the QP
    qp_symbol = 'X'
    #Add query point for density to graph
    qp = Atom(qp_symbol, position=xyz)
    #NB: Use lowest unoccupied atomic number for QP
    qp.number = max_atomic_num + 1
    atoms_.append(qp)
    #Create graph
    g = a2g.convert(atoms_)
    #Add density value to graph. This will be the target.
    #NB: Using the `y_relaxed` slot so OCP EnergyTrainer can be used
    target = rho.item()
    g.y_relaxed = target
    sid = "{}_{}".format(mp_id, idx)
    g.sid = sid
    g.db_id = mp_id
    return g
