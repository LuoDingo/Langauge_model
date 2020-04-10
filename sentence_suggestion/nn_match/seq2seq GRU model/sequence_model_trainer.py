import torch
import time
import math

def _epoch_time(start_time):
    total = time.time() - start_time
    return int(total/60), int(total%60)

def _default_init_weights(model):
    for name, param in model.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.01)

class TrainModel():

    def __init__(self,
                 model,
                 train_iterator,
                 val_iterator,
                 optimizer,
                 criterion,
                 output_dim,
                 weight_initializer=None):

        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.optimizer = optimizer
        self.criterion = criterion
        self.weight_initializer = weight_initializer
        self.output_dim = output_dim

    def train(self, clip):
        self.model.train()
        epoch_loss = 0
        for batch in self.train_iterator:
            keyword = batch.keywords
            # keyword = [keywords_len, batch_size]
            trg = batch.target
            # trg = [target_len, batch_size]
            trg_len = trg.shape[0]
            batch_size = trg.shape[1]

            self.optimizer.zero_grad()
            # prob: probability distribution over searching space given keywords for each batch
            prediction = self.model(keyword, trg)
            # prediction = [trg_len, batch_size, output_dim]

            # cut off the first token ([CLS]) and put batch and token distribution together
            prediction = prediction[1:].view((trg_len - 1)*batch_size, self.output_dim)
            # prediction = [(trg_len - 1)*batch_size, output_dim ]
            trg = trg[1:].view((trg_len - 1)*batch_size)
            # trg = [(trg_len - 1)*batch_size]

            loss = self.criterion(prediction, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(self.train_iterator)

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in self.val_iterator:
                keyword = batch.keywords
                # keyword = [keywords_len, batch_size]
                trg = batch.target
                # trg = [target_len, batch_size]
                trg_len = trg.shape[0]
                batch_size = trg.shape[1]

                # prob: probability distribution over searching space given keywords for each batch
                prediction = self.model(keyword, trg)
                # prediction = [trg_len, batch_size, output_dim]

                # cut off the first token ([CLS]) and put batch and token distribution together
                prediction = prediction[1:].view((trg_len - 1)*batch_size, self.output_dim)
                # prediction = [(trg_len - 1)*batch_size, output_dim ]
                trg = trg[1:].view((trg_len - 1)*batch_size)
                # trg = [(trg_len - 1)*batch_size]

                loss = self.criterion(prediction, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(self.val_iterator)

    def epoch(self, n_epochs, clip, model_name='transformer-lstm-model.pt'):

        # Initialize weights
        if self.weight_initializer==None:
            self.model.apply(_default_init_weights)
        else:
            self.model.apply(self.weight_initializer)
        # Keep track of the best model (the one with minimum validation loss)
        best_valid_loss = float('inf')
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = self.train(clip)
            valid_loss = self.evaluate()
            epoch_mins, epoch_secs = _epoch_time(start_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), model_name)
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
