# 🎑 Con2SES: Grid-Context Convolutional Model for Efficient Molecular Surface Construction from Neighboring Point Clouds
![](assets/surface.png)

## Dependencies
**Requirements**

- Python >= 3.9

- Packages
    ```bash
    pip install -r requirements.txt
    ```

## Training
To train the model on all image files:
```bash
# Con2SES-2D
sh scripts/2d/bash/train_on_sparse.sh

# Con2SES-3D
sh scripts/3d/bash/train_3d.sh
```

To eval the model on desired image files:
```bash
# Con2SES-2D
python scripts/2d/test.py

# Con2SES-3D
sh scripts/3d/bash/test_3d.sh
```
