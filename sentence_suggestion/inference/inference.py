import torch
from torch.nn.functional import log_softmax

from .model_blueprint import NNModel

import math
import csv


# The likelihood is computed as the sum of negative log probability of tokens
# (larger is more probable).
# Thus, longer sentences tend to have lower likelihoods as it has more negative
# values to add. We need to offset this disadvantage by dividing the sum by
# its length times penalty term for short sentences.
SHORT_LENGTH_PENALTY = 1.5


class SearchSpace():
    """
    This class stores the searching space (sentence candidates mapped into
    PyTorch tensors) and a model to compute the probability of those candidates
    given keywords. In its initialization phase, it downloads the searching
    space and constructs the seaching space. This operation may take a while.
    """
    def __init__(self,
                 input_vocab_path,
                 output_vocab_path,
                 model_path,
                 device_type,
                 sentence_candidates_path):
        self.model = NNModel(
            input_vocab_path=input_vocab_path,
            output_vocab_path=output_vocab_path,
            model_path=model_path,
            device_type=device_type
        )
        self.search_space = self._construct_search_space(
                                sentence_candidates_path
                            )


    def _construct_search_space(self, sentence_candidates_path):
        with open(sentence_candidates_path, newline='', encoding='utf-8') as f:
            # load sentence candidates (list of strings); skip header
            sentence_candidates = list(csv.reader(f))[1:]
        search_space = []
        for sen in sentence_candidates:
            # sen = ['..sentence..']
            sen = sen[0].lower().split()
            # add <cls> and <eos> token
            sen = [self.model.output_field.init_token] + sen + \
                  [self.model.output_field.eos_token]
            # convert tokens into their ids in the vocabulary
            enc_sen = [self.model.output_field.vocab.stoi[token] for token in sen]
            # convert list into tensor
            sen_tensor = torch.LongTensor(enc_sen).to(self.model.device)
            # sentence_tensor = [sentence_len]
            sen_tensor = sen_tensor.unsqueeze(1)
            # sentence_tensor = [sentences_len, 1]
            search_space.append(sen_tensor)
        return search_space


    def search_kbest(self, keywords, k):
        """Searches for the sentence suggestions in the searching space

        Parameters
        ----------
        keywords : str
            A bunch of keywords entered by users. keywords are expected to be
            separated by white spaces, e.g. 'what time movie'.
        k : int
            The number of sentences to be fetched from the sentence candidates in
            the searching space.

        Return
        ------
        suggestions : list of tuples
            The most probable k number of sentences found in the search space.
            It contains sentences with their probabilities, which are ordered
            by probability in descending order.

        Example
        -------
        >>> search_kbest('what time movie', 2)
        [('and what day would you like to see it ?', 0.0664),
         ('hi , can i get movie tickets here ?', 0.0662)]

        """
        # gotta discuss the format of keywords upon integration
        # tokenize keywords and make them lower case
        if isinstance(keywords, str):
            keywords = keywords.lower().split(' ')
        elif isinstance(keywords, list):
            keywords = [token.lower() for token in keywords]
        # add <cls> and <eos> token
        keywords = [self.model.input_field.init_token] + keywords + \
                   [self.model.input_field.eos_token]
        # convert tokens into their ids in the vocabulary
        enc_keywords = [self.model.input_field.vocab.stoi[token] for token in keywords]
        # convert list into tensor
        keywords_tensor = torch.LongTensor(enc_keywords).to(self.model.device)
        # keywords_tensor = [keywords_len]
        keywords_tensor = keywords_tensor.unsqueeze(1)
        # keywords_tensor = [keywords_len, 1]
        # Encodes keywords
        with torch.no_grad():
            # must pass keyword length for the sake of encoder
            keyword_len = torch.tensor(
                            [keywords_tensor.shape[0]]
                          ).to(self.model.device)
            # encode keywords
            encoder_output = self.model.encoder(keywords_tensor, keyword_len)
        # store the probability of sentences in target space
        prob_candidates = torch.zeros((1, len(self.search_space)),
                                      dtype=torch.float64)
        # prob_sen = [1, len(search_space)]
        # iterate through searching space
        for id, candidate in enumerate(self.search_space):
            # candidate = [candidate_len, 1]
            hidden = encoder_output
            candidate_len = candidate.shape[0]
            log_prob = 0
            # iterate through candidate sentence
            for i in range(1, candidate_len):
                # generates prob dist for the next word given current token
                with torch.no_grad():
                    # generate prob distribution of next word given inputs and previous word
                    cond_prob_dist, hidden = self.model.decoder(candidate[i-1],
                                                                hidden)
                    # cond_prob_dist = [1, output_dim]
                cond_prob_dist = cond_prob_dist.squeeze(0)
                # cond_prob_dist = [output_dim]
                log_prob_dist = log_softmax(cond_prob_dist, dim=0)
                # get the probability of next token given by the model
                log_prob += log_prob_dist[candidate[i]]
            # store log probability, it's divided by len^2 to offset sentence length
            prob_candidates[0, id] = log_prob/(candidate_len**SHORT_LENGTH_PENALTY )
        # get the top k prediction
        topk = torch.topk(prob_candidates, k, dim=1)
        # topk function returns (tensor(values), tensor(indices of tensors))
        # get indices of topk sentences with their probabilities
        topk_indices = tuple(zip(topk[0].tolist()[0], topk[1].tolist()[0]))
        # topk_indices = ((prob, sentence_idx), ..., (prob, sentence_idx))
        # convert tensors into strings
        topk_sentences = []
        for idx in topk_indices:
            # idx = (log prob, id)
            sen_tensor = self.search_space[idx[1]]
            # sentence_tensor = [sentence_len, 1]
            sen_tensor = sen_tensor.view(-1)
            # sentence_tensor = [sentence_len]
            # convert token ids into strings
            # sentence_tensor[0][1:-2] to remove <cls> and <eos> tokens
            sen = [self.model.output_field.vocab.itos[id] for id in sen_tensor[1:-1]]
            # store sentence and its probability
            topk_sentences.append((' '.join(sen), math.exp(idx[0])))
        return topk_sentences
