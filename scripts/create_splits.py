import argparse
from tqdm import tqdm
import numpy as np
from utils.splits import (
  write_split,
  split_val_test,
  get_elements,
  sort_by_id
)
from utils.loaders import load_structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


#IDs that failed to converge with aiida relax protocol
DROP_IDS = [
  'mp-10',       #As2
  'mp-894',      #ZnPt
  'mp-165',      #Si4
  'mp-1894',     #CW
  'mp-8883',     #AsGa
  'mp-1225811',  #Cu2S
  'mp-560025',   #Ag4S2
  'mp-570744',   #As4Si3
  'mp-570377',   #As4Sn3
  'mp-10759',    #GeSe
  'mp-1071269',  #Hg3Te3
  'mp-630528',   #In4S4
  'mp-1018019',  #Pb2S2
  'mp-13051',    #Si2Sr4
]

def main(args):

    seed = args.seed
    data_ids_fp = args.data_ids_fp
    split_sizes = args.split_sizes
    num_train, num_val, num_test = [int(x) for x in split_sizes.split(":")]
    out_dir = args.out_dir
    max_num_atoms = args.max_num_atoms
    min_num_atoms = args.min_num_atoms
    max_cell_volume = args.max_cell_vol
    assert data_ids_fp , "Must specify file path for IDs"

    #set seed for platform-specific determinism
    np.random.seed(seed)

    #Drop IDs with convergence problems
    print("Ad-hoc dropping these IDs: {}".format(DROP_IDS))

    #Load map of element combinations to IDs, and IDs to formulas
    key2ids = {}
    id2formula = {}
    id2num_atoms = {}
    id2vol = {}
    all_elems = set()
    with open(data_ids_fp, 'r') as fh:
        lines = fh.readlines()

    print("Filtering candidate list:")
    print("\t min number of atoms >= {}".format(min_num_atoms))
    print("\t max number of atoms <= {}".format(max_num_atoms))
    print("\t (conventional) cell volume <= {} (Ang^3)".format(max_cell_volume))
    for line in tqdm(lines):
        _, mp_id, formula = line.strip().split(" ")
        if mp_id in DROP_IDS:
            continue
        elems = get_elements(formula)
        struct = load_structure(mp_id)
        #Skip IDs we don't have a structure for
        if not struct:
            continue
        #Convert structure from primitive to conventional
        sga = SpacegroupAnalyzer(struct)
        conv_struct = sga.get_conventional_standard_structure()
        #Restrict range of number of atoms and prefer choices with
        #small cell volume to limit grid sizes.
        num_atoms = conv_struct.num_sites
        vol = conv_struct.volume
        if (   num_atoms > max_num_atoms or num_atoms < min_num_atoms
            or vol > max_cell_volume   ):
            continue
        #Track elements, num sites, volumes, formulas
        all_elems = all_elems.union(elems)
        id2num_atoms[mp_id] = num_atoms
        id2vol[mp_id] = vol
        id2formula[mp_id] = formula
        #Group formulas by their consistuent elements
        #Make a unique string per combo to key on
        key = " ".join(sorted(elems)) #alphabetize
        if key2ids.get(key):
            key2ids[key].append(mp_id)
        else:
            key2ids[key] = [mp_id]

    #Select one ID per key so that splits have disjoint combinations
    key2id = {}
    key2atom_counts = {}
    key2vol = {}
    for key, mp_ids in key2ids.items():
        #Randomly select one combination
        mp_id = np.random.choice(mp_ids, size=1)[0]
        key2id[key] = mp_id
        key2atom_counts[key] = id2num_atoms[mp_id]
        key2vol[key] = id2vol[mp_id]

    print("Number of keys after filtering: {}".format(len(key2id)))
    print()

    unaries   = [k for k in key2id.keys() if len(k.split(" ")) == 1]
    binaries  = [k for k in key2id.keys() if len(k.split(" ")) == 2]
    ternaries = [k for k in key2id.keys() if len(k.split(" ")) == 3]
    num_unary = len(unaries)
    num_binary = len(binaries)
    num_ternary = len(ternaries)
    print("\t num unary/binary/ternary = {}/{}/{}".format(
        num_unary, num_binary, num_ternary)
    )
    print()

    ### DEFINE SPLITS ###
    #
    # NB: Generalization to new combinations of elements is of special interest.
    #     Step 1: filter on number of atoms and cell volume,
    #     Step 2: Group structures by combination.
    #             Sample one structure per combination.
    #     Step 3: Assign the training split as:
    #             Train: All unaries and `num_train` random binaries
    #     Step 4: Val and test are then sampled from the remainder
    #             Val: `num_val` random binaries
    #             Test: two versions
    #                * Up to `num_test` random binaries
    #                * Up to `num_test`  random ternaries

    #Take all unaries for train
    unaries_train = unaries
    print("Unaries selected for training:")
    print("\t{}".format(unaries_train))
    print()

    ##Sample binaries for train
    binaries_train = sorted(
      np.random.choice(binaries, size=num_train, replace=False).tolist()
    )
    print("Binaries selected for training:")
    print("\t{}".format(binaries_train))
    print()

    #Check for any elements not represented in train set
    train_keys = unaries_train + binaries_train
    elems_train = set([y for x in train_keys for y in x.split(" ")])
    elems_not_in_train = all_elems.difference(elems_train)
    print("These elements not in train set: {}".format(elems_not_in_train))

    binaries_remaining = [
        b for b in binaries if b not in binaries_train
    ]
    ternaries_remaining = ternaries
    num_binaries_remaining = len(binaries_remaining)
    num_ternaries_remaining = len(ternaries_remaining)
    print("{} binaries remaining".format(num_binaries_remaining))
    print("{} ternaries remaining".format(num_ternaries_remaining))
    print()

    #Randomly select set of val/test binaries
    k2 = min(num_binaries_remaining, num_val+num_test)
    binaries_val_test = np.random.choice(
        binaries_remaining, size=k2, replace=False
    ).tolist()

    #Partition val/test binaries
    binaries_val   = binaries_val_test[:num_val]
    binaries_test  = binaries_val_test[num_val:]

    #Randomly select test ternaries
    k3 = min(num_ternaries_remaining, num_test)
    ternaries_test = np.random.choice(
        ternaries_remaining, size=k3, replace=False
    ).tolist()

    def create_split_name(keys, split):
        num_unary   = sum([len(k.split(" ")) == 1 for k in keys])
        num_binary  = sum([len(k.split(" ")) == 2 for k in keys])
        num_ternary = sum([len(k.split(" ")) == 3 for k in keys])
        if num_unary > 0 and num_binary > 0 and num_ternary > 0:
            split_name = '{}_unary_{}_binary_{}_ternary_{}'.format(
                split, num_unary, num_binary, num_ternary
            )
        elif num_unary > 0 and num_binary > 0 and num_ternary == 0:
            split_name = '{}_unary_{}_binary_{}'.format(
                split, num_unary, num_binary
            )
        elif num_unary == 0 and num_binary > 0 and num_ternary == 0:
            split_name = '{}_binary_{}'.format(split, num_binary)
        elif num_unary == 0 and num_binary == 0 and num_ternary > 0:
            split_name = '{}_ternary_{}'.format(split, num_ternary)
        elif num_unary == 0 and num_binary > 0 and num_ternary > 0:
            split_name = '{}_binary_{}_ternary_{}'.format(
                split, num_binary, num_ternary
            )
        split_name += '-no-ox'
        return split_name

    def write_split_(keys, split_name, out_dir):
        ids = [key2id[k] for k in keys]
        formulas = [id2formula[x] for x in ids]
        write_split(split_name, out_dir, ids, formulas)

    #Name splits
    train_split_name = create_split_name(train_keys, "train")
    binaries_val_split_name   = create_split_name(binaries_val, "val")
    binaries_test_split_name  = create_split_name(binaries_test, "test")
    ternaries_test_split_name = create_split_name(ternaries_test, "test")

    #Write splits
    write_split_(train_keys,     train_split_name,          out_dir)
    write_split_(binaries_val,   binaries_val_split_name,   out_dir)
    write_split_(binaries_test,  binaries_test_split_name,  out_dir)
    write_split_(ternaries_test, ternaries_test_split_name, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_ids_fp', type=str, default='',
        help='File path of data IDs to load')
    parser.add_argument('--out_dir', type=str,
        help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed for process')
    parser.add_argument('--split_sizes', type=str, default='25:5:100',
        help='Requested sizes of train/val/test. Exact match not guaranteed')
    parser.add_argument('--max_num_atoms', type=int, default=5,
        help='Limit max number of atoms per structure.')
    parser.add_argument('--min_num_atoms', type=int, default=2,
        help='Limit min number of atoms per structure.')
    parser.add_argument('--max_cell_vol', type=float, default=64,
        help='Limit min number of atoms per structure (Ang**3).')
    args = parser.parse_args()
    main(args)

