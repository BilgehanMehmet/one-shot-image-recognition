# Installation
```
git clone https://github.com/BilgehanMehmet/one-shot-image-recognition.git

cd one-shot-image-recognition
```

# Install datasets
## Omniglot 
```
git clone https://github.com/brendenlake/omniglot.git
cd omniglot/python 
unzip images_background.zip
unzip images_evaluation.zip
```

## AT&T Dataset
```
wget ftp://ftp.uk.research.att.com/pub/data/att_faces.zip
```

# Directory overview
```
--one-shot-learning directory
		main.py
		--docker
		--networks
			--siamese
				--base
				--mnist
				--omniglot
		--models
		--notebooks
		--pre_trained_weights
		--datasets
			--images_background
			--image_validation
			--images_evaluation
		--configs
		--bin
```
# Train

Run `main.py` to train the default one-shot learning model (Siamese Convolutional Neural Network) on the Omniglot dataset. Note that `main.py` calls the data loader, `loader.py`. For tuning purposes, `mode`, `n_examples`, `path` and `normalise` flag can all be modified in the `main.py` file. The path of trained weights can also be changed in the `filepath` variable of the function `train` in `main.py`. 

```
python3 main.py
```

You can also train the model on the ORL Database of Faces (AT&T Dataset) by running `model_orl.py`. The dataset contains 40 subjects, with 10 images per subject. We split the training and validation sets as follows: 30 subjects were used to train the model, and the remaining 10 subjects were used to generate one-shot trials. Of the 30 subjects, we had two cases, where in Case 1 the model was trained on 2 images per person, and 8 images were used for validation per person; and in Case 2, it was trained on 4 images per subject and 6 images were used per subject for validation purposes. The model can be trained by running:

```
python3 model_orl.py
```

Similarly, the data loader arguments can be modified during training and testing.


More information on the model architecture, dataset pre-processing and training details of one-shot learning can be found in the report provided for this project.




