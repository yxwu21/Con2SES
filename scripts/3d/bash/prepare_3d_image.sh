python scripts/prepare_3d_image_dataset.py \
    --mode gen_image \
    --abs_dat_paths "datasets/benchmark_data_0.9/*/*/*.dat" \
    --rel_dataset_path "datasets/benchmark_data_0.5_all" \
    --image_dataset_path "datasets/benchmark_3d_image_0.9" \
    --thread_num 16
