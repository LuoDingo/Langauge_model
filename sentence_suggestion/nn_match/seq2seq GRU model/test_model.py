from inference import SearchSpace


DATAPATH = 'C:\\path\\Masked Corpus'
MODEL_NAME = 'seq2seq-multilayer-gru.pt'


MASKED_TEXT = Field(
                sequential=True,
                init_token = '<sos>',
                eos_token = '<eos>',
                tokenize=lambda x: x.split(),
             )


TARGET_TEXT = Field(
                sequential=True,
                init_token = '<sos>',
                eos_token = '<eos>',
                tokenize=lambda x: x.split(),
             )


fields = [('id', None), ('keywords', MASKED_TEXT), ('target', TARGET_TEXT)]


train, val, test = TabularDataset.splits(
                            path=DATAPATH,
                            train='train.csv',
                            validation='val.csv',
                            test='test.csv',
                            format='csv',
                            skip_header=True,
                            fields=fields
                    )


MASKED_TEXT.build_vocab(train)
TARGET_TEXT.build_vocab(train)


EMB_DIM=256
ENC_INPUT_DIM=len(MASKED_TEXT.vocab)
DEC_INPUT_DIM=len(TARGET_TEXT.vocab)
OUTPUT_DIM=DEC_INPUT_DIM
N_LAYER=4
HID_DIM=1024
DROPOUT=0.3


model = seq2seq_multilayer_gru.Seq2Seq(
              enc_input_dim=ENC_INPUT_DIM,
              dec_input_dim=DEC_INPUT_DIM,
              emb_dim=EMB_DIM,
              enc_hid_dim=HID_DIM,
              dec_hid_dim=HID_DIM,
              n_layers=N_LAYER,
              output_dim=OUTPUT_DIM,
              device=device,
              dropout=DROPOUT
         ).to(device)


TARGET_SPACE = [
    'how many tickets would you like ?',
    'great i will reserve your seats',
    'yes that would be great',
    'yes please ! nine tickets please ',
    'i have purchased your tickets for you',
    'which theater do you want to go to ?',
    'and what day would you like to see it ?',
    'hi , can i get movie tickets here ?',
    'you are welcome goodbye ',
    'please repeat your last message',
]


KEYWORDS = 'how many tickets you ?'
K = 3


space = SearchSpace(
    model=model,
    model_name='seq2seq-multilayer-gru.pt',
    keywords_field=MASKED_TEXT,
    trg_field=TARGET_TEXT,
    device=device,
    target_sentences=TARGET_SPACE,
)
print(space.search_kbest(keywords=KEYWORDS, k=K))

# >> [('i have purchased your tickets for you', 0.047864415859750986),
#     ('how many tickets would you like ?', 0.02542865538029127),
#     ('great i will reserve your seats', 0.02403501413823161)]
