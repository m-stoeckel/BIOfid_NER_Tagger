from hyperopt import hp

from TokenLimitedColumnDataset import TokenLimitedColumnCorpus
from flair.embeddings import *
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.training_utils import EvaluationMetric

device = torch.device('cuda:0')

print('###############\n'
      '### Loading ###\n'
      '###############')
print('### Dataset ###')
corpus: Corpus = TokenLimitedColumnCorpus(
    data_folder='resources/data/',
    train_file='train.conll',
    test_file='test.conll',
    column_format={0: 'text', 1: 'pos', 2: 'lemma', 3: 'ner', 4: 'abstract', 5: 'metaphor'},
    min_sentence_length=5,
    max_sentence_length=50
)
tag_type = 'ner'

print('### Statistics ###')
log.info(corpus.obtain_statistics(tag_type=tag_type))

print('### Embeddings ###')
path = '/resources/nlp/embeddings/flair/'
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    StackedEmbeddings(embeddings=[
        # self-trained embeddings
        ##

        # pre-trained embeddings
        WordEmbeddings("de"),
        # FastTextEmbeddings("/resources/nlp/embeddings/latin/form/cc.la.300.vec"),
        BytePairEmbeddings(
            language='de', dim=300,
            cache_dir=path
        ),
        BytePairEmbeddings(
            language='la', dim=300,
            cache_dir=path
        ),
        FlairEmbeddings(
            'de-forward'
            # , pooling='min'
        ),
        FlairEmbeddings(
            'de-backward'
            # , pooling='min'
        ),
    ])
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[128, 256, 512])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[5e-3, 1e-3, 5e-4, 1e-4])
search_space.add(Parameter.MIN_LEARNING_RATE, hp.choice, options=[1e-7])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.OPTIMIZER, hp.choice, options=[torch.optim.Adam])
search_space.add(Parameter.EMBEDDINGS_STORAGE_MODE, hp.choice, options=["gpu"])

print('#################\n'
      '### Searching ###\n'
      '#################')
param_selector = SequenceTaggerParamSelector(
    corpus,
    tag_type,
    'resources/results_0',
    evaluation_metric=EvaluationMetric.MICRO_F1_SCORE
)
param_selector.optimize(search_space)

# model = 'BIOfid_02.12.19'
# tagger: SequenceTagger = SequenceTagger(
#     embeddings=embeddings,
#     tag_dictionary=corpus.make_tag_dictionary(tag_type=tag_type),
#     tag_type=tag_type,
#     hidden_size=256,
#     rnn_layers=1
# )
# print(tagger)
#
# trainer: ModelTrainer = ModelTrainer(tagger, corpus)
# trainer.train(
#     f'resources/models/{model}_{os.environ["CUDA_VISIBLE_DEVICES"]}',
#     learning_rate=0.1,
#     mini_batch_size=32,
#     patience=3,
#     use_amp=True,
#     monitor_train=False,
#     monitor_test=True,
#     embeddings_storage_mode='gpu'
# )

# print('###############\n'
#       '### Tagging ###\n'
#       '###############')
# out_path = f'system/{model}/'
# os.makedirs(out_path, exist_ok=True)
#
# found_tags = 0
# tq = tqdm(glob('resources/data/background_processed/*.conll'))
# for file in tq:
#     file_name = os.path.split(file)[1]
#     found_tags += process_file(tagger, file,
#                                os.path.join(out_path, file_name.replace('.conll', '.ann')))
#     tq.set_postfix(Tags=found_tags)
#
# print('##################\n'
#       '### Evaluating ###\n'
#       '##################')
# system(f'python3 evaluate.py ner gold/test_1.1 system/{model}/ | tee resources/models/{model}/eval_results.txt')
