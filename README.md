# Description
Classifies images and recordings of people.
Images are classified using small convolution network and 
audio classification is performed by training two Gaussian Mixture Models, 
one model for each target and non-target classs.


# Installation steps
```shell
python3.12 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```
# Data structure
Training data must be in following directories:
`data/training/target`
`data/training/non_target`

Validation data must be in following directories:
`data/validation/target`
`data/validation/non_target`

Evaluation audio files must be in folder `./data/eval`.


# Training models 
Please refer to image_training.ipynb for CNN training.

Training GMM for audio classification:
```shell
python3.12 --augment-audio --train-gmm
```
This saves trained gmm weights, means and covariances to `models/` folder.

# Usage with trained models
### Audio classification
 Prints predictions to stdout.
Example usage:
```shell
python3.12 main.py --predict > gmm_audio
```

### Image classification
Allows to specify image data path. 
Example usage:
```shell
python3.12 main.py --img-data-path  > image_cnn
```

