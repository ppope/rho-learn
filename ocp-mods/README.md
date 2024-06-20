# Modifications to OCP codebase

We forked the `ocp` repo at the following commit for training/evaluating models
* https://github.com/Open-Catalyst-Project/ocp/commit/a8ac36d75fc60d5b6d54b4019502941eb868fed0


### Training

We use the `EnergyTrainer` to train on density values.

We use LMDB files to train in the same format as used by OCP. See `create_lmdb.py` for more details.


### Evaluation

To predict density values for new test samples **without** creating LMDBs, use the custom loader to read density data from HDF5 files in `ocp/hdf5_dataset.py`.

To install this file in your `ocp` fork,
* copy the file to `ocp/ocpmodels/datasets`
* on L11 edit the (absolute) path to `rho-learn/utils` for your system
* reinstall ocp into your environment with `pip install -e .`

Example:
```
MP_ID="mp-129"
python ocp/main.py \
    --mode predict \
    --config-yml configs/SCN.yml \
    --checkpoint models/model.pt \
    --run-dir predictions \
    --identifier "$MP_ID" \
    --task.dataset="hdf5" \
    --dataset.test.metadata_file=metadata/scf.txt \
    --dataset.test.mp_ids="$MP_ID" \
    --dataset.test.max_neigh=12 \
    --dataset.test.radius=12 \
    --dataset.test.num_elements=83 \
    --optim.eval_batch_size=200 \
    --num-gpus 1 \
    --num-nodes 1
```


### Other changes

The following changes to the `ocp` codebase were made. These may or may not be necessary for your workflow.


Support string IDs for predictions. Useful for keying on Materials Project IDs:
```
diff --git a/ocpmodels/trainers/energy_trainer.py b/ocpmodels/trainers/energy_trainer.py
index 08e5ac4..bb02eb1 100644
--- a/ocpmodels/trainers/energy_trainer.py
+++ b/ocpmodels/trainers/energy_trainer.py
@@ -146,9 +146,17 @@ class EnergyTrainer(BaseTrainer):
                 )

             if per_image:
-                predictions["id"].extend(
-                    [str(i) for i in batch[0].sid.tolist()]
-                )
+                #phil: ad-hoc fix for list[str] sids rather than torch.tensor[int] sids.
+                #      torch.Tensors do not support string values
+                #      I use SIDs of the form e.g. "<MP-ID>_<SAMPLE-ID>"
+                if isinstance(batch[0].sid, list):
+                    predictions["id"].extend(
+                        [str(i) for i in batch[0].sid]
+                    )
+                else:
+                    predictions["id"].extend(
+                        [str(i) for i in batch[0].sid.tolist()]
+                    )
                 predictions["energy"].extend(
                     out["energy"].cpu().detach().numpy()
                 )
```

Read number of samples from length key rather than entries key. Entries key also tracks non-data entries like length and stats
```
diff --git a/ocpmodels/datasets/lmdb_dataset.py b/ocpmodels/datasets/lmdb_dataset.py
index c2020f0..b1cacc1 100644
--- a/ocpmodels/datasets/lmdb_dataset.py
+++ b/ocpmodels/datasets/lmdb_dataset.py
@@ -69,9 +69,12 @@ class LmdbDataset(Dataset):
         else:
             self.metadata_path = self.path.parent / "metadata.npz"
             self.env = self.connect_db(self.path)
+            length = pickle.loads(
+                self.env.begin().get("length".encode("ascii"))
+            )
             self._keys = [
                 f"{j}".encode("ascii")
-                for j in range(self.env.stat()["entries"])
+                for j in range(length)
             ]
             self.num_samples = len(self._keys)

```
