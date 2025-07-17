Here is the revised `README.md` with the conda environment option removed. Only manual pip-based installation is included now:

---

````markdown
# YouCan’tManufactureANeRF

This project explores geometric understanding and labeling from CAD-derived meshes to train Neural Radiance Fields (NeRFs). It includes tools for generating mesh-based point clouds, extracting geometric features like curvature, and labeling mesh segments for supervision.

---

## 🚀 Getting Started

This guide walks you through:

1. Installing all dependencies
2. Generating mesh files
3. Sampling point clouds from meshes
4. Labeling sampled point clouds using geometric information

---

## 🧩 Installation (Manual via pip)

Install the required Python packages using pip:

```bash
pip install \
    torch torchvision torchaudio \
    pytorch3d \
    numpy \
    trimesh \
    opencv-python \
    imageio imageio-ffmpeg \
    matplotlib \
    scikit-image \
    scikit-learn \
    tqdm \
    pandas \
    plyfile \
    open3d \
    PyYAML \
    pythonocc-core
````

> ✅ If you encounter issues installing `pythonocc-core`, try using conda:
>
> ```bash
> conda install -c conda-forge pythonocc-core
> ```

Optional visualization/debugging tools:

```bash
pip install pyvista vtk
```

---

## 🔧 Pipeline Overview

1. **Generate Meshes** (e.g. STEP/STL)
2. **Sample Point Clouds** using curvature-based sampling
3. **Label Point Clouds** by matching points to mesh faces

---

## 📁 Directory Structure

Your working directory should look like this:

```
project/
├── data/
│   ├── meshes/             # Raw CAD/mesh files (e.g. .stl, .step)
│   ├── sampled_pcds/       # Sampled points from meshes
│   ├── labeled_pcds/       # Ground truth labeled point clouds
├── generate_mesh_from_brep.py   # Sampling script
├── sample_mesh.py     # Labeling script
├── label_points_via_brep.py     # Labeling script
└── README.md
```

---

## 📦 Step-by-Step Usage

### 1. Prepare Mesh Files

Visit the [Fusion 360 Gallery Dataset GitHub page](https://github.com/AutodeskAILab/Fusion360GalleryDataset/tree/master) to get the `.stp` files.

Then run the script:

```bash
python generate_mesh_from_brep.py
```

The script will:

* Iterate through all `.stp` files in the `input_dir`
* Convert each one to a binary `.stl` file
* Save the mesh to `output_dir`
* Skip any mesh that already exists

---

### ⚙️ Mesh Precision

You can adjust the mesh quality by editing these parameters in the script:

```python
linear_deflection=0.025  # Smaller = more accurate mesh
angular_deflection=0.025
```

---

## 🆚 Default vs. Custom Meshes

| Option         | Resolution          | Notes                                  |
| -------------- | ------------------- | -------------------------------------- |
| Default `.stl` | Low                 | Included in the dataset                |
| Custom `.stl`  | Arbitrary precision | Generated with this script from `.stp` |

Place your CAD mesh files (`.stl` or `.step`) into the `data/meshes/` folder:

```bash
mkdir -p data/meshes
# Copy your STEP/STL files into this directory
```

---

### 2. Sample Point Clouds

Use the provided sampling script:

```bash
python sample_mesh.py
```

This will:

* Sample high-curvature edges and uniform surface points
* Output `.npy` files in `data/sampled_pcds/` containing:

  * XYZ coordinates
  * Normals
  * Face indices (if available)
  * Placeholder fields

Example output:

```
data/sampled_pcds/object1_processed.npy
```

---

### 3. Label Sampled Point Clouds

Once you have point clouds, label them using the face structure of the original CAD mesh:

```bash
python label_points_via_brep.py
```

This script will:

* Load each `.npy` from `sampled_pcds`
* Match each point to its closest face (based on distance and planarity)
* Output labeled point clouds in `data/labeled_pcds/`
  
---

### ✅ Full Example

```bash
# Step 1: Prepare your mesh files
cp *.stl data/meshes/

# Step 2: Sample them
python sample_mesh.py data/meshes data/sampled_pcds

# Step 3: Label them
python label_points_via_brep.py data/sampled_pcds data/labeled_pcds
```

---

## 🔍 Notes

* Curvature is computed using differential geometry approximations over k-NN neighborhoods.
* The labeling method relies on CAD surfaces and tolerance-based matching.
* Outputs can be extended with edge segmentation, curvature stats, or exported to `.ply` for visualization.
---

## 📄 License

[MIT License](LICENSE)

---

