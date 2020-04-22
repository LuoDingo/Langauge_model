"""
A module to construct the sentence searching space and provides a function
to search for the k number of most probable sentences given keywords.
The structure is the following:
inference/
    - inference.py: stores the searching space and provides search function in
                    the space.
    - model_blueprint.py: stores the same values of parameters of a neural
                          network model used during the training.
    - seq2seq_multilayer_gru_with_pad: a neural network model to compute the
                                       conditional probability of sentences
                                       given keywords.
"""
from .inference import SearchSpace

__all__ = [
    "SearchSpace"
]
