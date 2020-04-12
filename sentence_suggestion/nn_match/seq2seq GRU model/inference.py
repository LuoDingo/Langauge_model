import torch
from torch.nn.functional import log_softmax
import math


class SearchSpace():


    def __init__(self,
                 model,
                 model_name,
                 keywords_field,
                 trg_field,
                 device,
                 target_sentences):
        self.model = model
        # load model
        if device.type=='cpu':
            self.model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(model_name))
        self.KEYWORD_FIELD = keywords_field
        self.TRG_FIELD = trg_field
        self.device = device
        self.search_space = self._construct_search_space(target_sentences)


    def _construct_search_space(self, target_sentences):
        search_space = []
        for sentence in target_sentences:
            sentence = sentence.lower().split()
            # add <cls> and <eos> token
            sentence = [self.TRG_FIELD.init_token] + sentence + [self.TRG_FIELD.eos_token]
            # convert tokens into their ids in the vocabulary
            encoded_sentence = [self.TRG_FIELD.vocab.stoi[token] for token in sentence]
            # convert list into tensor
            sentence_tensor = torch.LongTensor(encoded_sentence).unsqueeze(1).to(self.device)
            # sentence_tensor = [sentences_len, 1]
            search_space.append(sentence_tensor)
        return search_space


    def search_kbest(self, keywords, k):
        # make sure we don't update model
        self.model.eval()
        # gotta discuss the format of keywords upon integration
        # tokenize keywords and make them lower case
        if isinstance(keywords, str):
            keywords = keywords.lower().split(' ')
        elif isinstance(keywords, list):
            keywords = [token.lower() for token in keywords]
        # add <cls> and <eos> token
        keywords = [self.KEYWORD_FIELD.init_token] + keywords + [self.KEYWORD_FIELD.eos_token]
        # convert tokens into their ids in the vocabulary
        encoded_keywords = [self.KEYWORD_FIELD.vocab.stoi[token] for token in keywords]
        # convert list into tensor
        keywords_tensor = torch.LongTensor(encoded_keywords).unsqueeze(1).to(self.device)
        # keywords_tensor = [keywords_len, 1]
        # Encodes keywords
        with torch.no_grad():
            encoder_output = self.model.encoder(keywords_tensor)
        # store the probability of sentences in target space
        prob_candidates = torch.zeros((1, len(self.search_space)), dtype=torch.float64)
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
                    cond_prob_dist, hidden = self.model.decoder(candidate[i-1], hidden)
                    # cond_prob_dist = [1, output_dim]
                cond_prob_dist = cond_prob_dist.squeeze(0)
                # cond_prob_dist = [output_dim]
                log_prob_dist = log_softmax(cond_prob_dist, dim=0)
                # get the probability of next token given by the model
                log_prob += log_prob_dist[candidate[i]]
            # store log probability, it's divided by len^2 to offset sentence length
            prob_candidates[0, id] = log_prob/(candidate_len**1.5)
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
            sentence_tensor = self.search_space[idx[1]]
            # sentence_tensor = [sentence_len, 1]
            sentence_tensor = sentence_tensor.view(-1)
            # sentence_tensor = [sentence_len]
            # convert token ids into strings
            # sentence_tensor[0][1:-2] to remove <cls> and <eos> tokens
            sentence = [self.TRG_FIELD.vocab.itos[id] for id in sentence_tensor[1:-1]]
            # store sentence and its probability
            topk_sentences.append((' '.join(sentence), math.exp(idx[0])))
        return topk_sentences
