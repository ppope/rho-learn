import os
import string
import numpy as np
from ase.data import chemical_symbols


def sort_by_id(ids):
    return sorted(ids, key=lambda x: int(x.split("-")[1]))


def check_disjoint(train, val, test):
    assert not set(train).intersection(set(val))
    assert not set(train).intersection(set(test))
    assert not set(test).intersection(set(val))


def write_split(split, out_dir, ids, formulas):
    print("Writing {}".format(split))
    fp = os.path.join(out_dir, '{}_ids.txt'.format(split))
    with open(fp, 'w') as fh:
        for i,f in zip(ids,formulas):
            f = f.replace(" ", "")
            fh.write("{} {}\n".format(i,f))


def split_val_test(elems2id, train_ids, test_n, val_n,
    out_dir, id2formula, num_elems=2
):
    remaining_ids = [
        v for k,v in elems2id.items() if (
            len(k.split(" ")) == num_elems and v not in train_ids
        )
    ]
    #split IDs train/val/test
    N = len(remaining_ids)
    all_inds = np.arange(N)
    rand_inds = np.random.choice(all_inds, size=N, replace=False)
    #val
    val_inds = rand_inds[test_n:(test_n + val_n)]
    val_ids = [remaining_ids[x] for x in val_inds]
    val_ids = sort_by_id(val_ids)
    val_formulas = [id2formula[x] for x in val_ids]
    #test
    test_inds = rand_inds[:test_n]
    test_ids = [remaining_ids[x] for x in test_inds]
    test_ids = sort_by_id(test_ids)
    test_formulas = [id2formula[x] for x in test_ids]
    #check disjointness
    check_disjoint(train_ids, val_ids, test_ids)
    #Write
    if num_elems == 2:
        tag = 'binary'
    elif num_elems == 3:
        tag = 'ternary'
    val_split_name =  'val_{}_{}'.format(tag, val_n)
    test_split_name =  'test_{}_{}'.format(tag, test_n)
    write_split(val_split_name,  out_dir, val_ids,  val_formulas)
    write_split(test_split_name, out_dir, test_ids, test_formulas)


def count_atoms_from_formula(formula, elems):
    """
    Counts atoms by their occurence in a formula, e.g. MnS2 -> 3
    """
    f = formula
    num_elems = len(elems)
    ##Normalize string to extract counts
    #Replace elements with space
    #Important: must replace longer strings first
    for e in sorted(elems, reverse=True):
        f = f.replace(e, " ")
    counts = [int(x) for x in f.split(" ") if x]
    #adjust counts for supressed "1"s in notation
    while len(counts) < num_elems:
       counts.append(1)
    total = sum(counts)
    return total


def get_elements(formula, max_num_elements=3):
    """
    Parse elements from formula
    """
    f = formula
    ##Normalize string to extract elements
    ##Drop parens?... not necessary for OCP formulas
    #f = f.replace("(", " ").replace(")", " ")
    #Drop digits
    for x in string.digits:
        f = f.replace(x, " ")
    #Add space between elements. Two cases:
    #   (1) Upper-Upper, e.g. SV
    #   (2) Lower-Upper e.g. NaCl
    fl = list(f)
    add_spaces = []
    for i,(x,y) in enumerate(zip(fl, fl[1:])):
        x_is_upper = x in string.ascii_uppercase
        y_is_upper = y in string.ascii_uppercase
        if (    (     x_is_upper and y_is_upper)
             or ( not x_is_upper and y_is_upper) ):
            add_spaces.append(i+1)
    #Add the spaces:
    #Important: must reverse list so indices don't change
    for i in reversed(add_spaces):
        f = f[:i] + " " + f[i:]
    #Finally, split and drop spaces
    elems = set([x for x in f.split(" ") if x])
    #Check elements are valid
    for e in elems:
        assert e in chemical_symbols
    #OCP bulks have number of elements < 3
    num_elems = len(elems)
    assert num_elems <= max_num_elements
    #alphabetize list
    elems = list(elems)
    elems = sorted(elems)
    return elems

