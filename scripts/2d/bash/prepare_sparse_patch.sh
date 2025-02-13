python -m scripts.2d.prepare_sparse_dataset \
    --mode gen_patch \
    --image_dataset_path "datasets/benchmark_image_0.9_sparse" \
    --image-npz-paths "datasets/benchmark_image_0.9_sparse/*/*/*/*.npz" \
    --patch-size 128