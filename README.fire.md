git clone https://github.com/PeizhuoLi/neural-blend-shapes.git
scoop install micromamba
micromamba activate base
micromamba install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install tensorboard
pip install tdqm
pip install chumpy
pip install "numpy<1.24.0"
python demo.py --animated_bvh=1 --obj_output=0 --pose_file=./eval_constant/sequences/greeting.npy --obj_path=./eval_constant/meshes/maynard.obj 
cd blender_scripts
blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.glb
# Test with triangulated mesh
python demo.py --animated_bvh=0 --obj_output=0 --pose_file=./eval_constant/sequences/house-dance.npy --obj_path=./eval_constant/meshes/fire-fleur.obj --normalize=1
cd blender_scripts
blender -b -P nbs_fbx_output.py -- --input ../demo --output ../demo/output.glb