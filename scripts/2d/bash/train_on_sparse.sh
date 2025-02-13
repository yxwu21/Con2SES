OUTPUT_FOLDER=outputs/train/$(date +%Y%m%d_%H%M%S)
WANDB_API_KEY=YOUR_WANDB_KEY

python -m scripts.2d.train --trainer.dataset-path datasets/benchmark_image_0.9_sparse/patches_index_sz128_split.json \
    --trainer.patch-size 128 \
    --trainer.output-folder $OUTPUT_FOLDER \
    --trainer.train-lr 1e-4 \
    --trainer.train-num-steps 60000 \
    --trainer.train-batch-size 32 \
    --trainer.eval-batch-size 64 \
    --trainer.save-and-eval-every 5000 \
    --trainer.num-workers 4 \
    --trainer.probe-radius-upperbound 1.5 \
    --trainer.probe-radius-lowerbound -1.5 \
    --trainer.use-sparse-surface-dataset \
    --trainer.sparse-train-dataset-config.patch-slice-axis z \
    --trainer.sparse-train-dataset-config.random-rotate \
    --trainer.sparse-test-dataset-config.patch-slice-axis z \
    --model.kernel-sizes 1 5 7 \
    --wandb.name 2d_train_sparse_light \
    --wandb.project surface \
    --wandb.api-key $WANDB_API_KEY \
    --slurm.mode slurm \
    --slurm.slurm_mem 64G \
    --slurm.slurm_job_name 2d \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 12 \
    --slurm.gpus-per-node 1 \
    --slurm.node_list laniakea