# run this cell, then restart the runtime before continuing

import os, csv
from itertools import compress
import warnings

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import transformers
from datasets import load_dataset, load_metric

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from argparse import ArgumentParser


def get_conll_data(split: str = 'train', 
                   limit: int = None, 
                   dir: str = None) -> dict:
    """Load CoNLL-2003 (English) data split.
    Loads a single data split from the 
    [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) 
    (English) data set.
    Args:
        split (str, optional): Choose which split to load. Choose 
            from 'train', 'valid' and 'test'. Defaults to 'train'.
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned.
        dir (str, optional): Directory where data is cached. If set to 
            None, the function will try to look for files in '.conll' folder in home directory.
    Returns:
        dict: Dictionary with word-tokenized 'sentences' and named 
        entity 'tags' in IOB format.
    Examples:
        Get test split
        >>> get_conll_data('test')
        Get first 5 observations from training split
        >>> get_conll_data('train', limit = 5)
    """
    assert isinstance(split, str)
    splits = ['train', 'valid', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.conll')
    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'
    
    file_path = os.path.join(dir, f'{split}.txt')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'

    # read data from file.
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ' ')
        for row in reader:
            data.append([row])

    sentences = []
    sentence = []
    entities = []
    tags = []

    for row in data:
        # extract first element of list.
        row = row[0]
        # TO DO: move to data reader.
        if len(row) > 0 and row[0] != '-DOCSTART-':
            sentence.append(row[0])
            tags.append(row[-1])        
        if len(row) == 0 and len(sentence) > 0:
            # clean up sentence/tags.
            # remove white spaces.
            selector = [word != ' ' for word in sentence]
            sentence = list(compress(sentence, selector))
            tags = list(compress(tags, selector))
            # append if sentence length is still greater than zero..
            if len(sentence) > 0:
                sentences.append(sentence)
                entities.append(tags)
            sentence = []
            tags = []
            
   
    if limit is not None:
        sentences = sentences[:limit]
        entities = entities[:limit]
    
    return {'sentences': sentences, 'tags': entities}

## loading here for testing purposes - data will be loaded thru DataModule for the Model
train_data = get_conll_data(split='train',dir='conll2003')
val_data = get_conll_data(split='valid',dir='conll2003')
test_data = get_conll_data(split='test',dir='conll2003')

print('train data size',len(train_data['sentences']))
print('val data size',len(val_data['sentences']))
print('test data size',len(test_data['sentences']))

print(train_data['sentences'][4])

print(train_data['tags'][4])

#model_checkpoint = 'bert-base-multilingual-uncased'
model_checkpoint  = 'distilbert-base-uncased'
transformer_model = transformers.AutoModel.from_pretrained(model_checkpoint)
transformer_tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)
transformer_config = transformers.AutoConfig.from_pretrained(model_checkpoint)  

class NERDataSet(Dataset):
    """Generic NERDA DataSetReader"""
    
    def __init__(self, 
                examples, 
                tokenizer: transformers.PreTrainedTokenizer,
                tag_encoder: sklearn.preprocessing.LabelEncoder, 
                label_all_tokens: bool = False  
                ) -> None:
        """Initialize DataSetReader
        Initializes DataSetReader that prepares and preprocesses 
        DataSet for Named-Entity Recognition Task and training.
        Args:
            sentences (list): Sentences.
            tags (list): Named-Entity tags.
            transformer_tokenizer (transformers.PreTrainedTokenizer): 
                tokenizer for transformer.
            transformer_config (transformers.PretrainedConfig): Config
                for transformer model.
            max_len (int): Maximum length of sentences after applying
                transformer tokenizer.
            tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
                for Named-Entity tags.
            tag_outside (str): Special Outside tag.
        """
        self.sentences = examples['sentences']
        self.tags = examples['tags']
        self.tokenizer = tokenizer
        self.tag_encoder = tag_encoder
        self.label_all_tokens = label_all_tokens
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        tags = self.tags[item]
        # encode tags and sentence words
        tags = self.tag_encoder.transform(tags)
        tokenized_inputs = self.tokenizer(self.sentences[item], truncation=True, is_split_into_words=True)

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.  A word could be split into two or more tokens occasionally depending on the model tokenizer
            else:
                label_ids.append(tags[word_idx] if self.label_all_tokens else -100)
            previous_word_idx = word_idx

        tokenized_inputs["target_tags"] = label_ids
        return tokenized_inputs

### Define the tag scheme.  These tags are pre-defined in the CONLL data
def get_tag_scheme():
  tag_scheme = [
  'B-PER',
  'I-PER',
  'B-ORG',
  'I-ORG',
  'B-LOC',
  'I-LOC',
  'B-MISC',
  'I-MISC'
  ]
  tag_outside = 'O'
  tag_complete = [tag_outside] + tag_scheme
  return tag_complete

class NERDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 2):
        super().__init__()
   
        # Defining batch size of our data
        self.batch_size = batch_size
          
        # Defining num_workers
        self.num_workers = num_workers

        # Defining Tokenizers
        self.tokenizer = transformer_tokenizer

        self.label_pad_token_id = -100
  
    def prepare_data(self):
        self.train_data = get_conll_data(split='train',dir='conll2003')
        self.val_data = get_conll_data(split='valid',dir='conll2003')
        self.test_data = get_conll_data(split='test',dir='conll2003')

        self.tag_complete = get_tag_scheme()
        self.tag_encoder = sklearn.preprocessing.LabelEncoder()
        self.tag_encoder.fit(self.tag_complete)
  
    def setup(self, stage=None):
        # Loading the dataset
        self.train_dataset = NERDataSet(self.train_data, tokenizer=self.tokenizer, tag_encoder=self.tag_encoder, label_all_tokens=True)
        self.val_dataset = NERDataSet(self.val_data, tokenizer=self.tokenizer, tag_encoder=self.tag_encoder, label_all_tokens=True)
        self.test_dataset = NERDataSet(self.test_data, tokenizer=self.tokenizer, tag_encoder=self.tag_encoder, label_all_tokens=True)
  
    def custom_collate(self,features):
        label_name = "target_tags"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        
        batch = self.tokenizer.pad(  
            features,
            padding=True,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch[label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch    
        
    def train_dataloader(self):
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #return DataLoader(train_dataset, sampler=dist_sampler, batch_size=32) # For use in Multiple GPUs
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def val_dataloader(self):
         return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

class NERModel(torch.nn.Module):

    def __init__(self,
                  n_tags: int, dropout: float = 0.1, 
                 **kwargs):
    #def __init__(self, conf, **kwargs):   
        super().__init__()
 
        self.n_tags = n_tags
        self.dropout = dropout
        self.transformer = transformer_model
        # extract transformer name
        self.transformer_name = self.transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        self.transformer_config = transformer_config

        
        self.dropout = torch.nn.Dropout(dropout)
        self.tags = torch.nn.Linear(self.transformer_config.hidden_size, n_tags)
    
    def forward(self,  batch)-> torch.Tensor:
        """Model Forward Iteration
        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            
        Returns:
            torch.Tensor: predicted values.
        """        

        outputs = self.transformer(input_ids=batch['input_ids'], \
                         attention_mask=batch['attention_mask'])

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        
        # apply drop-out
        outputs = self.dropout(hidden_state)

        # outputs for all labels/tags
        outputs = self.tags(outputs)

        return outputs

## The main Pytorch Lightning module

class NERTokenClassifier(pl.LightningModule):

    def __init__(self, n_tags: int, learning_rate: float = 0.0001 * 8, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_tags = n_tags
        # Metrics
        self.metric = load_metric("seqeval")       
        self.model = NERModel(n_tags=self.n_tags)         

    def training_step(self, batch, batch_nb):
        target_tags = batch['target_tags']
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        
        # Compute active loss so as to not compute loss of paddings
        active_loss = batch['attention_mask'].view(-1) == 1

        active_logits = y_hat.view(-1, self.n_tags)
        active_labels = torch.where(
            active_loss,
            target_tags.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(target_tags)
        )

        # Only compute loss on actual token predictions
        loss = loss_fct(active_logits, active_labels)

        # logs
        self.log_dict({'train_loss':loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        target_tags = batch['target_tags']
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        
        # Compute active loss so as to not compute loss of paddings
        active_loss = batch['attention_mask'].view(-1) == 1

        active_logits = y_hat.view(-1, self.n_tags)
        active_labels = torch.where(
            active_loss,
           target_tags.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(target_tags)
        )

        # Only compute loss on actual token predictions
        loss = loss_fct(active_logits, active_labels)

        metrics = self.compute_metrics([y_hat,target_tags])

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log_dict({'val_loss':loss, 'val_f1':metrics['f1'], 'val_accuracy':metrics['accuracy'], 
                       'val_precision':metrics['precision'], 'val_recall':metrics['recall']}, prog_bar=True)
        return loss    

    def test_step(self, batch, batch_nb):
        target_tags = batch['target_tags']
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        # Compute active loss so as to not compute loss of paddings
        active_loss = batch['attention_mask'].view(-1) == 1

        active_logits = y_hat.view(-1, self.n_tags)
        active_labels = torch.where(
            active_loss,
            target_tags.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(target_tags)
        )

        # Only compute loss on actual token predictions
        loss = loss_fct(active_logits, active_labels)
        metrics = self.compute_metrics([y_hat,target_tags])
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log_dict({'test_loss':loss, 'test_f1':metrics['f1'], 'test_accuracy':metrics['accuracy'], 
                       'test_precision':metrics['precision'], 'test_recall':metrics['recall']}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        # fwd
        y_hat = self.model(batch)
        return {'logits':y_hat, 
                'target_tags':batch['target_tags'],
                'input_ids':batch['input_ids'],
                'attention_mask':batch['attention_mask']}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()), epochs=self.hparams.max_epochs),
        'interval': 'step'  # called after each training step
        } 
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=2000)
        #scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
        #scheduler = ReduceLROnPlateau(optimizer, patience=0, factor=0.2)

        return [optimizer], [scheduler]
        
       
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        #parser.add_argument('--drop_prob', default=0.2, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'train_val_data'), type=str)

        # training params (opt)
        parser.add_argument('--learning_rate', default=2e-5, type=float, help = "type (default: %(default)f)")
        return parser
    # ---------------------
    # EVALUATE PERFORMANCE
    # --------------------- 

    def compute_metrics(self,p):
      predictions, labels = p
      predictions = torch.argmax(predictions, dim=2)
      label_len = len(self.trainer.datamodule.tag_complete)
      label_list = self.trainer.datamodule.tag_encoder.inverse_transform(np.arange(label_len))
      
      # Remove ignored index (special tokens)
      true_predictions = [
          [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]
      true_labels = [
          [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]

      results = self.metric.compute(predictions=true_predictions, references=true_labels)
      return {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
          "accuracy": results["overall_accuracy"],
      }

# ------------------------
# TRAINING ARGUMENTS
# ------------------------
# these are project-wide arguments
root_dir = os.getcwd()
parent_parser = ArgumentParser(add_help=False)
parent_parser = pl.Trainer.add_argparse_args(parent_parser)

# each LightningModule defines arguments relevant to it
parser = NERTokenClassifier.add_model_specific_args(parent_parser,root_dir)

parser.set_defaults(
    #profiler='simple',
    deterministic=True,
    max_epochs=3,
    gpus=1,
    distributed_backend=None,
    fast_dev_run=False,
    model_load=False,
    model_name='best_model',
    n_tags = len(get_tag_scheme())
)

args, extra = parser.parse_known_args()

""" Main training routine specific for this project. """
# ------------------------
# 1 INIT LIGHTNING MODEL
# ------------------------
if (vars(args)['model_load']):
  model = NERTokenClassifier.load_from_checkpoint(vars(args)['model_name'])
else:  
  model = NERTokenClassifier(**vars(args))
print('n_tags',model.n_tags)
# ------------------------
# 2 CALLBACKS of MODEL
# ------------------------

# callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0,
    patience=3,
    verbose=True,
    mode='min',
    strict=True,
)

lr_monitor = LearningRateMonitor(logging_interval='step')

checkpoint_callback = ModelCheckpoint(
     monitor='val_loss',
     #dirpath='my/path/',
     filename='conll-ner-epoch{epoch:02d}-val_loss{val_loss:.2f}',
     auto_insert_metric_name=False
)

# ------------------------
# 3 INIT TRAINER
# ------------------------
trainer = Trainer.from_argparse_args(args,
    callbacks=[early_stop,lr_monitor,checkpoint_callback]
    )    

seed_everything(42, workers=True)
conll_dm = NERDataModule()

# ------------------------
# 4 START TRAINING
# ------------------------
trainer.fit(model,conll_dm)
trainer.validate()
trainer.test()

## This will run the predict_step using the predict_dataloader.  The predict_step is made to return both the logits and labels for the Test data
conll_dm = NERDataModule()
conll_dm.prepare_data()
conll_dm.setup()
val_dataloader = conll_dm.val_dataloader()
predict_with_labels = trainer.predict(dataloaders=val_dataloader)
## flatten the Labels and Predictions after choosing the argmax of the each of the logits
predictions_flat, labels_flat, input_ids_flat = [], [], []
for index, batch in enumerate(predict_with_labels):
  predictions, labels, input_ids = batch['logits'], batch['target_tags'], batch['input_ids']
  predictions_flat.extend(torch.argmax(predictions, dim=2))
  labels_flat.extend(labels)
  input_ids_flat.extend(input_ids.cpu())

metric = load_metric("seqeval")

label_list = conll_dm.tag_encoder.inverse_transform(np.arange(9))

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions_flat, labels_flat)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions_flat, labels_flat)
]
true_input_ids = [
    [i.item() for (i, l) in zip(id, label) if l != -100]
    for id, label in zip(input_ids_flat, labels_flat)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results

checkpoint_callback.best_model_path

ls -al lightning_logs/version_11/checkpoints

## Load the Model
model_eval = NERTokenClassifier.load_from_checkpoint(checkpoint_callback.best_model_path).eval()

text1 = ['This is Hugging Face Inc. based in New York City',"Germany's representative to the European Union's veterinary committee Werner Zwingmann said on Wednesday", 
         "the Commission's chief spokesman Nikolaus van der Pas told a news briefing."]
#text1 = [test_data['sentences'][15]]         
tokenized_inputs = transformer_tokenizer(text1, truncation=True, padding=True, return_tensors="pt",is_split_into_words=False)
logits = model_eval.model(tokenized_inputs)
label_list = conll_dm.tag_encoder.inverse_transform(np.arange(9))
predictions = torch.argmax(logits, dim=2)
### 101, 102 and 0 represent the CLS, SEP and padding tokens
output = [[(transformer_tokenizer.decode(ids), label_list[p]) for (p,ids) in zip(prediction,input_ids) if ids not in [101,102,0]] 
          for prediction, input_ids in zip(predictions,tokenized_inputs['input_ids'])]
print(output)

%reload_ext tensorboard
%tensorboard --logdir lightning_logs/

!ls -al 'lightning_logs/version_0/checkpoints/'

# device = "cuda:0"
# model = NERModel.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=2-step=2633.ckpt')
# #model = NERModel(n_tags = len(get_tag_scheme()),learning_rate=2e-5)
# model.freeze()

# conll_dm = NERDataModule(batch_size = len(test_data['tags']))
# #conll_dm = NERDataModule()
# conll_dm.prepare_data()
# conll_dm.setup()
# test_dataloader = conll_dm.test_dataloader()
# batch = next(iter(test_dataloader))

# predictions = model(batch)

predictions.shape

metric = load_metric("seqeval")
labels = batch['target_tags']
predictions = np.argmax(predictions.cpu(), axis=2)
label_list = conll_dm.tag_complete

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results

len(batch['attention_mask'])

#model.eval()
#model.freeze()
#test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 5, shuffle=False)

# I try this when Colab runs out of Cuda memory
torch.cuda.empty_cache()

!/opt/bin/nvidia-smi

!ps -aux|grep python

# This is the best way to free up GPU memory - kill the ipykernel process
!kill -9 1129



## Trying out the LR Find method in Pytorch Lightning.  This won't work for multi gpu situations.  Wasn't happy with the initial results of the Learning rate finder.
## This code won't work without defining bert_imdb variable
bert_ner = NERModel(transformer = transformer_model,
                    n_tags = len(tag_complete))
trainer = pl.Trainer(gpus=1, max_epochs=1, auto_lr_find=True)

# Run learning rate finder
lr_finder = trainer.fit(bert_ner)

# Results can be found in
lr_finder.results

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()


