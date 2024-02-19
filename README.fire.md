# README.fire

```bash
# Clone the repository using Git
git clone https://github.com/PeizhuoLi/neural-blend-shapes.git

scoop install miniconda

# Activate base environment (usually conda is automatically initiated and no need for 'conda activate base' if it's the default)
# But if needed, here is how to activate the base environment with Conda:
conda activate base

# Install pytorch, torchvision, torchaudio along with CUDA toolkit from the PyTorch channel
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -c nvidia

# Install other required Python packages using pip
pip install tensorboard
pip install tqdm
pip install chumpy
pip install "numpy<1.24.0"

# Running the demo script from the cloned repository
python demo.py --animated_bvh=1 --obj_output=0 --pose_file=./eval_constant/sequences/greeting.npy --obj_path=./eval_constant/meshes/maynard.obj

# Navigate into the blender_scripts directory within the cloned repository
cd blender_scripts

# Run the Blender script in background mode
blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.glb

# Run another instance of the demo script without clothing, as per the context given
python ../demo.py --animated_bvh=0 --obj_output=0 --pose_file=../eval_constant/sequences/house-dance.npy --obj_path=../eval_constant/meshes/fire-fleur.obj

# Again, run the Blender script in background mode with new input and output
blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.glb

# Custom
python ../demo.py --animated_bvh=0 --obj_output=0 --pose_file=../eval_constant/sequences/house-dance.npy --obj_path=../eval_constant/meshes/custom.obj
```