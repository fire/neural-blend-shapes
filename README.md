# Learning Skeletal Articulations with Neural Blend Shapes

![Python](https://img.shields.io/badge/Python->=3.8-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.8.0-Red?logo=pytorch)
![Blender](https://img.shields.io/badge/Blender-%3E=2.8-Orange?logo=blender)

This repository provides an end-to-end library for automatic character rigging, skinning, blend shapes generation, and a visualization tool.

It is based on our work [Learning Skeletal Articulations with Neural Blend [Shapes](https://peizhuoli.github.io/neural-blend-shapes/index.html), published in SIGGRAPH 2021.

<img src="https://peizhuoli.github.io/neural-blend-shapes/images/video_teaser.gif" slign="center">

## Prerequisites

Our code has been tested on Ubuntu 18.04. Before starting, please configure your Anaconda environment by

~~~bash
conda env create -f environment.yaml
conda activate neural-blend-shapes
~~~

Or you may install the following packages (and their dependencies) manually:

- pytorch 1.8
- tensorboard
- tqdm
- chumpy

Note the provided environment only includes the PyTorch CPU version for compatibility consideration.

## Quick Start

We provide a pre-trained model that is dedicated to biped characters. Download and extract the pre-trained model from [Google Drive](https://drive.google.com/file/d/1S_JQY2N4qx1V6micWiIiNkHercs557rG/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1y8iBqf1QfxcPWO0AWd2aVw) (9ras) and put the `pre_trained` folder under the project directory. Run

~~~bash
python demo.py --pose_file=./eval_constant/sequences/greeting.npy --obj_path=./eval_constant/meshes/maynard.obj
~~~

The nice greeting animation shown above will be saved in `demo/obj` as obj files. In addition, the generated skeleton will also be saved as `demo/skeleton.bvh` and the skinning weight matrix as `demo/weight.npy`. If you need the bvh file animated, you may specify `--animated_bvh=1`.

If you are interested in the traditional linear blend skinning (LBS) technique result generated with our rig, you can specify `--envelope_only=1` to evaluate our model only with the envelope branch.

We also provide several other meshes and animation sequences. Feel free to try their combinations!


### FBX Output (New!)

Now, you can output the animation as a single FBX file instead of a sequence of obj files! Do the following:

You must install Blender (>=2.80) to generate the FBX file. You may explore more options on the generated FBX file in the source code.

This code is contributed by [@huh8686](https://github.com/huh8686).

### Test on Customized Meshes

You may try to run our model with your meshes by pointing the `--obj_path` argument to the input mesh. Please ensure your mesh is triangulated and has a consistent upright and front facing orientation. Since our model requires the input meshes to be spatially aligned, please specify `--normalize=1`. Alternatively, you can try to scale and translate your mesh to align the provided `eval_constant/meshes/smpl_std.obj` without specifying `--normalize=1`.

### Evaluation

To reconstruct the quantitative result with the pre-trained model, download the test dataset from [Google Drive](https://drive.google.com/file/d/1RwdnnFYT30L8CkUb1E36uQwLNZd1EmvP/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1c5QCQE3RXzqZo6PeYjhtqQ) (8b0f), put the two extracted folders under `./dataset`, and run.

~~~bash
python evaluation.py
~~~


## Train from Scratch

We provide instructions for retraining our model.

You may need to reinstall the PyTorch CUDA version since the provided environment only includes the PyTorch CPU version.

You may download the training set from [Google Drive](https://drive.google.com/file/d/1RSd6cPYRuzt8RYWcCVL0FFFsL42OeHA7/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1J-hIVyz19hKZdwKPfS3TtQ) (uqub) and put the extracted folders under `./dataset` to train the model from scratch.

The training process contains two stages, each stage corresponding to one branch. To train in the first stage, please run

~~~bash
python train.py --envelope=1 --save_path=[path to save the model] --device=[cpu/cuda:0/cuda:1/...]
~~~

For the second stage, it is strongly recommended to use a pre-process to extract the blend shapes basis then start the training for much better efficiency by

~~~bash
python preprocess_bs.py --save_path=[same path as the first stage] --device=[computing device]
python train.py --residual=1 --save_path=[same path as the first stage] --device=[computing device] --lr=1e-4
~~~

## Blender Visualization

We provide a simple wrapper of Blender's Python API (>=2.80) to render 3D mesh animations and visualize skinning weight. The following code has been tested on Ubuntu 18.04 and macOS Big Sur with Blender 2.92.

Note that due to the limitation of Blender, you cannot run Eevee render engine with a headless machine. 

We also provide several arguments to control the behavior of the scripts. Please refer to the code for more details. To pass arguments to the Python script in Blender, please do the following:

~~~bash
blender [blend file path (optional)] -P [python script path] [-b (running at backstage, optional)] -- --arg1 [ARG1] --arg2 [ARG2]
~~~



### Animation

We provide a simple light and camera setting in `eval_constant/simple_scene.blend`. You may need to adjust it before using it. We use `ffmpeg` to convert images into video. Please make sure you have installed it before running. To render the obj files generated above, run

~~~bash
cd blender_script
blender ../eval_constant/simple_scene.blend -P render_mesh.py -b
~~~

The rendered per-frame image will be saved in `demo/images`, and the composited video will be saved as `demo/video.mov`. 

### Skinning Weight

Visualizing the skinning weight is an excellent check to see whether the model works as expected. We provide a script using Blender's built-in ShaderNodeVertexColor to visualize the skinning weight.

~~~bash
cd blender_script
blender -P vertex_color.py
~~~

You will see something similar to this if the model works as expected:

<img src="https://peizhuoli.github.io/neural-blend-shapes/images/skinning_vis.png" slign="center" width="50%">

Meanwhile, you can import the generated skeleton (in `demo/skeleton.bvh`) to Blender. For skeleton rendering, please refer to [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing).

## Acknowledgements

The code in `blender_scripts/nbs_fbx_output.py` is contributed by [@huh8686](https://github.com/huh8686).

The code in `meshcnn` is adapted from [MeshCNN](https://github.com/ranahanocka/MeshCNN) by [@ranahanocka](https://github.com/ranahanocka/).

The code in `models/skeleton.py` is adapted from [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing) by [@kfiraberman, [@PeizhuoLi](https://github.com/PeizhuoLi), and [@HalfSummer11](https://github.com/HalfSummer11).

The code in `dataset/smpl.py` is adapted from [SMPL](https://github.com/CalciferZh/SMPL) by [@CalciferZh](https://github.com/CalciferZh).

Some test models are taken from [SMPL](https://smpl.is.tue.mpg.de/en), [MultiGarmentNetwork](https://github.com/bharat-b7/MultiGarmentNetwork), and [Adobe Mixamo](https://www.mixamo.com).

## Citation

If you use this code for your research, please cite our paper:

~~~bibtex
@article{li2021learning,
  author = {Li, Peizhuo and Aberman, Kfir and Hanocka, Rana and Liu, Libin and Sorkine-Hornung, Olga and Chen, Baoquan},
  title = {Learning Skeletal Articulations with Neural Blend Shapes},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {40},
  number = {4},
  pages = {130},
  year = {2021},
  publisher = {ACM}
}
~~~

