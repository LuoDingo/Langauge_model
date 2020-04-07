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
                 weight_initializer=None):

        self.model = model
        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.optimizer = optimizer
        self.criterion = criterion
        self.weight_initializer = weight_initializer

    def train(self, clip):
        self.model.train()
        epoch_loss = 0
        for c, batch in numerate(self.train_iterator):
            print(f'{c}-th batch ruuning')
            keyword = batch.keywords
            trg = batch.target
            target_id = batch.id
            #target_id = [batch_size]

            self.optimizer.zero_grad()
            # prob: probability distribution over searching space given keywords for each batch
            prob_dist = self.model(keyword, trg)
            #pred = [batch_size, output_dim]

            loss = self.criterion(prob_dist, target_id)
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
                trg = batch.target
                target_id = batch.id

                pred_id = self.model(keyword, trg)
                #pred_id = [1, batch size]

                pred_id = pred_id.squeeze()
                #pred_id = [batch_size]

                loss = self.criterion(pred_id, target_id)
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
