# rho-learn

Code for "Towards Combinatorial Generalization for Catalysts: A Kohn-Sham Charge-Density Approach" (NeurIPS 2023)

Please consider citing our work if you find this repo useful:
```
@inproceedings{NEURIPS2023_be82bb4b,
 author = {Pope, Phillip and Jacobs, David},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {60585--60598},
 publisher = {Curran Associates, Inc.},
 title = {Towards Combinatorial Generalization for Catalysts: A Kohn-Sham Charge-Density Approach},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/be82bb4bf8333107b0fe430e1017831a-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

### Dataset

The complete dataset of raw DFT runs used in the paper may be found on Zenodo in the `aiida.tar.gz` file
* https://zenodo.org/records/11680781

Unfortunately we cannot host LMDB files used for training due to storage constraints. These LMDBs files take about 560 GB of disk space uncompressed. They can be recreated with `create_lmdbs.py. This may take several hours depending on how many CPUs you have available.


### Code

`ocp-mods` contains code and documentation for the modifications to the `ocp` codebase we used for training/evaluation

`configs` contains complete hyperparameters for the model used in the paper

`metadata` contains important metadata such as the unique IDs of structures used in the project, and also how these IDs are split into train/val/test.

`scripts` contains useful scripts for the project, such as creating LMDB files from the charge-density data generated by the DFT code.

`utils` contains project specific code, such as loading charge density files from `HDF5` format


### Model

Checkpoint file for the model used to report results in the the paper may be found on Zenodo in the `model.tar.gz` file.
