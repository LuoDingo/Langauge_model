import requests
from tqdm import tqdm
from time import sleep
import os


class DataLoader():
    """
    This class is used to download a pre-trained neural network model from url.
    It stores model name and the link to a model.
    """
    def __init__(self):
        # url where model is stored
        self.model_url = 'https://www.dropbox.com/s/exxhu5p2st4nvau/seq2seq-multilayer-gru.pt?dl=1'
        # output file name
        self.model_path = 'nnmodel.pt'
        # chunk size for progress bar
        self.chunk_size = 2**20
        self._load()

    def _load(self):
        # if a model exists skip this process
        if not os.path.exists(self.model_path):
            # get model
            r  = requests.get(self.model_url, stream=True)
            # output model
            with open(self.model_path, 'wb') as f:
                size = int(r.headers.get('content-length'))
                task = 'Download NN model'
                # print progress bar
                with tqdm(total=size, unit=' data', desc=task) as pbar:
                    for chunk in r.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            pbar.update(len(chunk))
