# Langauge_model
Here is where the NLP Language models will be stored, in python

# Note (by Kei)
Beam-search is implemented in the following package: https://github.com/box-key/zen-corpora
`inference_model/inference.py` implements exhaustive serach (searching everything).

# Repository structure

* `dataset`: contains searching space and masked datasets used for training nn model.
* `sentence_suggestion/train_nn_models`: contains codes showing how to train model and to make dataset.
* `sentence_suggestion/inference`: contain codes to be used in application.
	* `inference.py`: encodes input and computes the conditional probability of each sentence in the searching space.
	* `loader.py`: loading pre-trained model from online storage.
	* `model_blueprint.py`: stores hyperparameters of a model used in inference.
* `keyword_match`: non deep learning sentence matching model.