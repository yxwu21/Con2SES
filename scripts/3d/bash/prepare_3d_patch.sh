python scripts/prepare_3d_image_dataset.py \
    --mode gen_patch \
    --image_dataset_path "datasets/benchmark_3d_image_0.9" \
    --image-npz-paths "datasets/benchmark_3d_image_0.9/*/*/*/*.pkl.gz" \
    --patch-size 64