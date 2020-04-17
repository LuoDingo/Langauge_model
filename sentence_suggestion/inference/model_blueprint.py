from .seq2seq_multilayer_gru_with_pad import Seq2Seq
import dill
import torch
import torchtext


EMB_DIM = 256
N_LAYER = 4
HID_DIM = 1024
DROPOUT = 0.3


class NNModel():
    """
    This class stores the detail of trained model stored as
    "seq2seq-multilayer-gru.pt". Since loading the model requires to give
    exactly the same parameters used for training, we need to store the blue
    print of a model.
    First, the model is initialized with exactly the same parameter values used
    during the training, then it loads model. It also store the vocabularies
    used during the training.
    """
    def __init__(self,
                 input_vocab_path,
                 output_vocab_path,
                 model_path,
                 device_type):
        # store torchtext's field objects
        self.input_field = self._load_dill(input_vocab_path)
        self.output_field = self._load_dill(output_vocab_path)
        self.device = torch.device(device_type)
        self.model = Seq2Seq(
              device=self.device,
              enc_input_dim=len(self.input_field.vocab),
              dec_input_dim=len(self.output_field.vocab),
              output_dim=len(self.output_field.vocab),
              emb_dim=EMB_DIM,
              enc_hid_dim=HID_DIM,
              dec_hid_dim=HID_DIM,
              n_layers=N_LAYER,
              dropout=DROPOUT
        ).to(self.device)
        self._load_model(model_path)
        # make sure we don't update model
        self.model.eval()
        self.decoder = self.model.decoder
        self.encoder = self.model.encoder


    # torchtext's version should be >= 0.5
    def _load_dill(self, path):
        with open(path, "rb") as f:
            obj = dill.load(f)
        return obj


    def _load_model(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
