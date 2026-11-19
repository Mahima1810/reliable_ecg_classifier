# Turing setup

**Status: not verified.** No Turing documentation was available in the development environment, and no scheduler, partition,
storage, quota or CUDA detail for Turing was confirmed. The scripts in `turing/` are templates. They stop with an error until
the values below are filled in and the file is marked verified.

## Information required before the first job
| Item | Needed for | Value |
|---|---|---|
| Scheduler (SLURM, PBS, LSF, or other) and submission command | `turing/train.sbatch.template` | TBD |
| Account / project name | job submission | TBD |
| Partition or queue with GPUs | job submission | TBD |
| GPU model, count per node, and architecture | precision choice (fp16 on sm_75 or Ampere; bf16 only on Ampere or newer) | TBD |
| Driver and CUDA runtime version | choosing the torch wheel (`environment/requirements.txt`) | TBD |
| Node-local scratch path and free-space rules | copying caches to node-local disk | TBD |
| Persistent storage path for data, runs and checkpoints | `ECG_DATA_ROOT`, `ECG_RUN_ROOT` | TBD |
| Storage quota (home and project) | dataset and run sizing | TBD |
| Wall-time, memory and job-count limits | `--time`, `--mem` | TBD |
| Internet access from compute nodes | whether datasets can be fetched at job time (should be no) | TBD |

## Note on the only GPU architecture reference found
A Turing-architecture GPU (RTX 2080 Ti, sm_75) is referenced in a different project. It does not support bf16 efficiently and
does not support flash-attention 2. If the Turing cluster uses the same hardware, fp16 mixed precision is the required choice.
This repository does not yet implement mixed precision; that is a remaining blocker (see README).

## Procedure once the values are known
1. Fill in `turing/env.sh` (copy from `turing/env.sh.example`) with the paths. Do not commit it.
2. Run `turing/setup_env.sh` on a login node to create the environment.
3. Run the data validation job (`turing/validate.sbatch.template`) and check the report.
4. Only after the owner approves the seed list and clinical settings, submit training.
5. Record the job ID, node, library versions and the commit hash with each run.
