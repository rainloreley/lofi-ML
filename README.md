# Lofi ML

This program will train a machine learning model to generate Lofi music

**This is just an experiment, don't use that music to study. It's shit**

Based on the work from [Skuldur](https://github.com/Skuldur/Classical-Piano-Composer) and [this article](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)

## Setup
1. Install and setup [Anaconda](https://www.anaconda.com) on your system

<details>
<summary>CPU version</summary>

2. Run `conda create -n tensorflow_env tensorflow`
3. Run `conda activate tensorflow_env`
</details>

<details>
<summary>GPU version</summary>

2. `conda create -n tensorflow_gpuenv tensorflow-gpu`
3. `conda activate tensorflow_gpuenv`
</details>
-----
4. Install missing dependencies (TBD, too lazy to do now)

## Install training MIDI files
Prepare a dataset of MIDI files you want to use for training.

Create a new folder called `trainingmidis` and copy the MIDIs into it.

You can find a set of Lofi MIDIs [here](https://www.kaggle.com/datasets/zakarii/lofi-hip-hop-midi?resource=download)

## Run training
`python3 main.py train`

Available parameters:
- `--epochs=`: Set the amount of epochs the model will run through (_Default: 200_)
- `--batchsize=`: Set the batch size for each epoch (_Default: 64_)

After the training, the program will find the best model and copy its weight file into the `bestmodels` directory.

During the training, the model will take checkpoints and save them to `modelcheckpoints/{ISO 8601 datetime}/`, so you can stop training whenever you like. This means the program won't find the best model, you have to do that manually.

## Generate music
`python3 main.py gen --weightfile={path}`

Replace `{path} with the path to your model weightfile`

Available parameters:
- `notes=`: Set the amount of notes to be generated (_Default: 100_)

## Debug
<details>

<summary>TensorFlow can't allocate enough resources (GPU only)</summary>

Run `export TF_FORCE_GPU_ALLOW_GROWTH='true'`

</details>

<details>
<summary>The model only generates one repeating note</summary>

Experiment with the amount of epochs and batch size for training. Loss values < 0.1 seem to work

</details>
