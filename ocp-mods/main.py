"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import copy
import logging

import submitit

from ocpmodels.common.flags import flags
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging()

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    #To get multi-node training on UMIACS clusters working
    os.environ["NCCL_IB_DISABLE"] = "1"
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()

    #Handle on-the-fly config for test predictions as special case of overrides
    test_args = {}
    pop_list = []
    for i,x in enumerate(override_args):
        test_arg_prefix =  "--dataset.test."
        if test_arg_prefix in x:
            #Warning: assumes test arg is given in form:
            #         "--dataset.test.arg_name=arg_val"
            arg_name, arg_val = x.split("=")
            arg_name = arg_name.replace(test_arg_prefix, "")
            test_args[arg_name] = arg_val
            pop_list.append(i)
    if pop_list:
        override_args = [
            x for i,x in enumerate(override_args) if i not in pop_list
        ]

    config = build_config(args, override_args)

    if test_args:
        assert test_args['mp_ids'], "No MP-ID specified!"
        train_d = config['dataset'][0]
        #Inherit normalization from train config
        test_args['normalize_labels'] = train_d.get('normalize_labels')
        test_args['target_mean'] = train_d.get('target_mean')
        test_args['target_std'] = train_d.get('target_std')
        #Fix arg types:
        test_args['max_neigh'] = int(test_args['max_neigh'])
        test_args['num_elements'] = int(test_args['num_elements'])
        test_args['radius'] = float(test_args['radius'])
        #Assume data for test predictions is never resampled
        test_args['p_high'] = 0.
        test_args['p_unif'] = 0.
        #Current OCP code does not allow different dataset classes
        #   to be loaded at the same time for different splits.
        #   We load all splits as HDF5 but set train and val to null.
        null_args = copy.deepcopy(test_args)
        null_args['mp_ids'] = ','
        dlist_new = [null_args, null_args, test_args]
        config['dataset'] = dlist_new
        #Make predictions sequential
        config['optim']['num_workers'] = 0

    if args.submit:  # Run on cluster
        slurm_add_params = config.get(
            "slurm", None
        )  # additional slurm arguments
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(
            folder=args.logdir / "%j", slurm_max_num_timeout=3
        )
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            #gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(
            f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
        )
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config)
