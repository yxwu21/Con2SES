DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/test_3d/$DATETIME
WANDB_API_KEY=YOUR_WANDB_KEY

python -m scripts.3d.test_3d \
    --trainer-ckpt-path outputs/train_3d/20241225_221835/model-5.pt \
    --trainer.dataset-path datasets/benchmark_3d_image_0.9/patches_index_sz64.json \
    --trainer.patch-size 64 \
    --trainer.output-folder $OUTPUT_FOLDER \
    --trainer.train-lr 1e-4 \
    --trainer.train-num-steps 60000 \
    --trainer.train-batch-size 4 \
    --trainer.eval-batch-size 4 \
    --trainer.save-and-eval-every 10000 \
    --trainer.num-workers 4 \
    --trainer.probe-radius-upperbound 1.5 \
    --trainer.probe-radius-lowerbound -1.5 \
    --model.kernel-sizes 1 5 7 \
    --model.model-type light \
    --wandb.name 3d_test \
    --wandb.project surface \
    --wandb.api-key $WANDB_API_KEY \
    --slurm.mode slurm \
    --slurm.slurm-partition YOUR_SLURM \
    --slurm.slurm-job-name 3d \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 6 \
    --slurm.gpus-per-node 1 \
    --slurm.mem 256GB \
    --slurm.node_list YOUR_NODE