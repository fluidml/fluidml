from dataclasses import dataclass
from datetime import datetime
import gzip
import math
import multiprocessing
import os
import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
from tokenizers import Tokenizer
from tokenizers.implementations import CharBPETokenizer
from tokenizers.processors import TemplateProcessing
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.metrics import bleu_score
from tqdm import tqdm

from transformer_model import Seq2SeqTransformer, Encoder, Decoder
from fluidml import Flow, Swarm
from fluidml.common import Task, Resource
from fluidml.flow import GridTaskSpec, TaskSpec
from fluidml.storage import LocalFileStore
from fluidml.common.logging import configure_logging


configure_logging()


def get_balanced_devices(count: Optional[int] = None,
                         use_cuda: bool = True) -> List[str]:
    count = count if count is not None else multiprocessing.cpu_count()
    if use_cuda and torch.cuda.is_available():
        devices = [f'cuda:{id_}' for id_ in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def set_seed(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)


class MyLocalFileStore(LocalFileStore):
    def __init__(self, base_dir: str):
        super().__init__(base_dir=base_dir)

        self._save_load_fn_from_type['torch'] = (self._save_torch, self._load_torch)
        self._save_load_fn_from_type['tokenizer'] = (self._save_tokenizer, self._load_tokenizer)

    @staticmethod
    def _save_torch(name: str, obj: Any, run_dir: str):
        torch.save(obj, f=os.path.join(run_dir, f'{name}.pt'))

    @staticmethod
    def _load_torch(name: str, run_dir: str) -> Any:
        return torch.load(os.path.join(run_dir, f'{name}.pt'))

    @staticmethod
    def _save_tokenizer(name: str, obj: Tokenizer, run_dir: str):
        obj.save(os.path.join(run_dir, f'{name}.json'))

    @staticmethod
    def _load_tokenizer(name: str, run_dir: str) -> Tokenizer:
        return Tokenizer.from_file(os.path.join(run_dir, f'{name}.json'))


@dataclass
class TaskResource(Resource):
    device: str
    seed: int


class DatasetLoading(Task):
    def __init__(self,
                 base_url: str,
                 data_split_names: Dict[str, List]):
        super().__init__()
        self.base_url = base_url
        self.data_split_names = data_split_names

    @staticmethod
    def download_and_extract_gz_from_url(url: str) -> List[str]:
        # download gz compressed data
        data_gz = requests.get(url=url)
        # decompress downloaded gz data to bytes object
        data_bytes = gzip.decompress(data_gz.content)
        # decode bytes object to utf-8 encoded str and convert to list by splitting on new line chars
        data = data_bytes.decode('utf-8').splitlines()
        return data

    def run(self):
        for split_name, files in self.data_split_names.items():
            dataset = {}
            for file_name in files:
                # create download url
                url = self.base_url + file_name
                language = file_name.split('.')[1]
                # download and parse data
                data = DatasetLoading.download_and_extract_gz_from_url(url=url)
                dataset[language] = data
            # save train-, valid- and test-data as json via local file store
            self.save(obj=dataset, name=f'{split_name}_data', type_='json')


class TokenizerTraining(Task):
    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train_tokenizer(self, data: List[str]):
        # initialize and train a tokenizer
        tokenizer = CharBPETokenizer()
        tokenizer.train_from_iterator(iterator=data,
                                      vocab_size=self.vocab_size,
                                      min_frequency=self.min_frequency,
                                      special_tokens=['<unk>', '<bos>', '<eos>', '<pad>'],
                                      show_progress=True)

        # add template rule to automatically add <bos> and <eos> to the encoding
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair=None,
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        return tokenizer

    def run(self, train_data: Dict[str, List[str]]):
        # train german tokenizer
        de_tokenizer = self.train_tokenizer(data=train_data['de'])

        # train english tokenizer
        en_tokenizer = self.train_tokenizer(data=train_data['en'])

        # save tokenizers
        self.save(obj=de_tokenizer, name='de_tokenizer', type_='tokenizer')
        self.save(obj=en_tokenizer, name='en_tokenizer', type_='tokenizer')


class DatasetEncoding(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def encode_data(data: Dict[str, List[str]],
                    src_tokenizer: Tokenizer,
                    trg_tokenizer: Tokenizer) -> List[Tuple[List[int], List[int]]]:

        src_encoded = src_tokenizer.encode_batch(data['de'])
        trg_encoded = trg_tokenizer.encode_batch(data['en'])
        return [(src.ids, trg.ids) for src, trg in zip(src_encoded, trg_encoded)]

    def run(self,
            train_data: Dict[str, List[str]],
            valid_data: Dict[str, List[str]],
            test_data: Dict[str, List[str]],
            de_tokenizer: Tokenizer,
            en_tokenizer: Tokenizer):

        train_encoded = DatasetEncoding.encode_data(train_data, de_tokenizer, en_tokenizer)
        valid_encoded = DatasetEncoding.encode_data(valid_data, de_tokenizer, en_tokenizer)
        test_encoded = DatasetEncoding.encode_data(test_data, de_tokenizer, en_tokenizer)

        self.save(obj=train_encoded, name='train_encoded', type_='json')
        self.save(obj=valid_encoded, name='valid_encoded', type_='json')
        self.save(obj=test_encoded, name='test_encoded', type_='json')


class BatchCollator:
    def __init__(self, de_pad_idx: int, en_pad_idx: int, device: str):
        self.de_pad_idx = de_pad_idx
        self.en_pad_idx = en_pad_idx
        self.device = device

    def __call__(self, batch):
        de_batch, en_batch = [], []
        for de_item, en_item in batch:
            de_batch.append(torch.tensor(de_item, dtype=torch.long))
            en_batch.append(torch.tensor(en_item, dtype=torch.long))
        de_batch = pad_sequence(de_batch, batch_first=True, padding_value=self.de_pad_idx).to(self.device)
        en_batch = pad_sequence(en_batch, batch_first=True, padding_value=self.en_pad_idx).to(self.device)
        return de_batch, en_batch


class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[List[int], List[int]]]):
        self.data = data

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src_sample = self.data[idx][0]
        trg_sample = self.data[idx][1]
        return src_sample, trg_sample

    def __len__(self):
        return len(self.data)


class Training(Task):
    def __init__(self,
                 hid_dim: int,
                 enc_layers: int,
                 dec_layers: int,
                 enc_heads: int,
                 dec_heads: int,
                 enc_pf_dim: int,
                 dec_pf_dim: int,
                 enc_dropout: float,
                 dec_dropout: float,
                 learning_rate: float,
                 clip_grad: float,
                 train_batch_size: int,
                 valid_batch_size: int,
                 num_epochs: int):
        super().__init__()

        # transformer model parameters
        self.hid_dim = hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        self.enc_pf_dim = enc_pf_dim
        self.dec_pf_dim = dec_pf_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout

        # optimizer parameters
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad

        # dataloader and training loop parameters
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_epochs = num_epochs

    def _init_training(self, input_dim: int, output_dim: int, src_pad_idx: int, trg_pad_idx: int):
        """ Initialize all training components.
        """

        # initialize the encoder and decoder block
        enc = Encoder(input_dim,
                      self.hid_dim,
                      self.enc_layers,
                      self.enc_heads,
                      self.enc_pf_dim,
                      self.enc_dropout,
                      self.resource.device)

        dec = Decoder(output_dim,
                      self.hid_dim,
                      self.dec_layers,
                      self.dec_heads,
                      self.dec_pf_dim,
                      self.dec_dropout,
                      self.resource.device)

        # initialize the full transformer sequence to sequence model
        model = Seq2SeqTransformer(enc, dec, src_pad_idx, trg_pad_idx, self.resource.device).to(self.resource.device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # initialize the loss criterion
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        return model, optimizer, criterion

    def _train_epoch(self, model, iterator, optimizer, criterion):
        """ Train loop to iterate over batches
        """
        model.train()

        epoch_loss = 0

        for i, (src, trg) in enumerate(iterator):

            optimizer.zero_grad()

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # [batch size * trg len - 1, output dim]
            trg = trg[:, 1:].contiguous().view(-1)             # [batch size * trg len - 1]

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)

            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    @staticmethod
    def validate_epoch(model, iterator, criterion):
        """ Validation loop to iterate over batches
        """
        model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for src, trg in iterator:

                output, _ = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)  # [batch size * trg len - 1, output dim]
                trg = trg[:, 1:].contiguous().view(-1)             # [batch size * trg len - 1]

                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def _train(self, model, train_iterator, valid_iterator, optimizer, criterion):
        """ Train loop.
        """

        best_valid_loss = float('inf')
        best_model = None

        for epoch in tqdm(range(self.num_epochs)):

            start_time = datetime.now()
            train_loss = self._train_epoch(model, train_iterator, optimizer, criterion)
            valid_loss = Training.validate_epoch(model, valid_iterator, criterion)
            end_time = datetime.now()

            # if the current validation loss is below the previous best, update the best loss and
            # save the new best model.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = model.state_dict()
                self.save(obj=model.state_dict(), name='best_model', type_='torch')
                self.save(obj={'epoch': epoch,
                               'valid_loss': best_valid_loss}, name='best_model_metric', type_='json')

            print(f'Epoch: {epoch + 1:02} | Time: {end_time - start_time}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        assert best_model is not None
        return best_model, best_valid_loss

    def run(self,
            train_encoded: List[Tuple[List[int], List[int]]],
            valid_encoded: List[Tuple[List[int], List[int]]],
            de_tokenizer: Tokenizer,
            en_tokenizer: Tokenizer):
        set_seed(self.resource.seed)

        # instantiate the collate fn for the dataloader
        batch_collator = BatchCollator(de_pad_idx=de_tokenizer.token_to_id('<pad>'),
                                       en_pad_idx=en_tokenizer.token_to_id('<pad>'),
                                       device=self.resource.device)

        # instantiate train and validation datasets using a pytorch's Dataset class
        train_dataset = TranslationDataset(data=train_encoded)
        valid_dataset = TranslationDataset(data=valid_encoded)

        # instantiate train and validation dataloader
        train_iterator = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True,
                                    collate_fn=batch_collator)
        valid_iterator = DataLoader(valid_dataset, batch_size=self.valid_batch_size, shuffle=False,
                                    collate_fn=batch_collator)

        input_dim = de_tokenizer.get_vocab_size()
        output_dim = en_tokenizer.get_vocab_size()
        src_pad_idx = de_tokenizer.token_to_id('<pad>')
        trg_pad_idx = en_tokenizer.token_to_id('<pad>')

        # instantiate all training components
        model, optimizer, criterion = self._init_training(input_dim=input_dim,
                                                          output_dim=output_dim,
                                                          src_pad_idx=src_pad_idx,
                                                          trg_pad_idx=trg_pad_idx)

        # train the model on the training set and evaluate after every epoch on the validation set
        self._train(model=model,
                    train_iterator=train_iterator,
                    valid_iterator=valid_iterator,
                    optimizer=optimizer,
                    criterion=criterion)


class ModelSelection(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _select_best_model_from_sweeps(training_results: List[Dict]) -> Dict:
        config = None
        best_valid_loss = float('inf')
        for sweep in training_results:
            if sweep['result']['best_model_metric']['valid_loss'] <= best_valid_loss:
                best_valid_loss = sweep['result']['best_model_metric']['valid_loss']
                config = sweep['config']
        return config

    def run(self, reduced_results: List[Dict]):

        # select the best run config by comparing model performances from different parameter sweeps
        # on the validation set
        best_run_config = self._select_best_model_from_sweeps(training_results=reduced_results)

        self.save(obj=best_run_config, name='best_run_config', type_='json')


class Evaluation(Task):
    def __init__(self, test_batch_size: int):
        super().__init__()

        self.batch_size = test_batch_size

    def _init_model(self, train_config: Dict, input_dim: int, output_dim: int,
                    src_pad_idx: int, trg_pad_idx: int) -> nn.Module:
        """ Initialize the model and its components.
        """

        enc = Encoder(input_dim,
                      train_config['hid_dim'],
                      train_config['enc_layers'],
                      train_config['enc_heads'],
                      train_config['enc_pf_dim'],
                      train_config['enc_dropout'],
                      self.resource.device)

        dec = Decoder(output_dim,
                      train_config['hid_dim'],
                      train_config['dec_layers'],
                      train_config['dec_heads'],
                      train_config['dec_pf_dim'],
                      train_config['dec_dropout'],
                      self.resource.device)

        model = Seq2SeqTransformer(enc, dec, src_pad_idx, trg_pad_idx, self.resource.device).to(self.resource.device)
        return model

    def translate_sentence(self, src_encoded, bos_idx, eos_idx, model, max_len=50):
        """ Translate an encoded sentence.
        """

        model.eval()

        src_tensor = torch.LongTensor(src_encoded).unsqueeze(0).to(self.resource.device)
        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indices = [bos_idx]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(self.resource.device)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indices.append(pred_token)
            if pred_token == eos_idx:
                break

        return trg_indices[1:]

    def calculate_bleu(self, data_encoded, en_tokenizer, model, max_len=50):
        """ Calculate the bleu score on the test set.
        """

        trgs = []
        pred_trgs = []
        bos_idx = en_tokenizer.token_to_id('<bos>')
        eos_idx = en_tokenizer.token_to_id('<eos>')

        for src, trg in tqdm(data_encoded):

            pred_trg = self.translate_sentence(src, bos_idx, eos_idx, model, max_len)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trg_decoded = en_tokenizer.decode(pred_trg)
            pred_trgs.append(pred_trg_decoded.split())

            trg_decoded = en_tokenizer.decode(trg)
            trgs.append(trg_decoded.split())

        return bleu_score(pred_trgs, trgs)

    def run(self, best_run_config: Dict):
        set_seed(self.resource.seed)

        # load the best model, test-data and the tokenizers based on the previously selected best run config
        model_state_dict = self.load(name='best_model', task_name='Training', task_unique_config=best_run_config)
        test_encoded = self.load(name='test_encoded', task_name='DatasetEncoding', task_unique_config=best_run_config)
        de_tokenizer = self.load(name='de_tokenizer', task_name='TokenizerTraining', task_unique_config=best_run_config)
        en_tokenizer = self.load(name='en_tokenizer', task_name='TokenizerTraining', task_unique_config=best_run_config)

        # instantiate the batch collator
        batch_collator = BatchCollator(de_pad_idx=de_tokenizer.token_to_id('<pad>'),
                                       en_pad_idx=en_tokenizer.token_to_id('<pad>'),
                                       device=self.resource.device)

        # instantiate the test dataset
        test_dataset = TranslationDataset(data=test_encoded)

        # instantiate the test dataloader
        test_iterator = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=batch_collator)

        input_dim = de_tokenizer.get_vocab_size()
        output_dim = en_tokenizer.get_vocab_size()
        src_pad_idx = de_tokenizer.token_to_id('<pad>')
        trg_pad_idx = en_tokenizer.token_to_id('<pad>')

        # instantiate the transformer model
        model = self._init_model(train_config=best_run_config['Training'],
                                 input_dim=input_dim,
                                 output_dim=output_dim,
                                 src_pad_idx=src_pad_idx,
                                 trg_pad_idx=src_pad_idx)
        model.load_state_dict(model_state_dict)

        # instantiate the loss criterion
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        # evaluate the model on the test set -> calculate the test set loss and perplexity
        test_loss = Training.validate_epoch(model=model, iterator=test_iterator, criterion=criterion)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        # calculate the model's bleu score on the test set
        bleu = self.calculate_bleu(test_encoded, en_tokenizer, model)
        print(f'BLEU score = {bleu * 100:.2f}')


def main():
    current_dir = os.path.abspath('')
    base_dir = os.path.join(current_dir, 'seq2seq_experiments')

    num_workers = 1
    force = None  # choices [selected, all, None]
    task_to_execute = None
    use_cuda = True
    seed = 1234

    dataset_loading_params = {'base_url': 'https://raw.githubusercontent.com/multi30k/dataset/'
                                          'master/data/task1/raw/',
                              'data_split_names': {'train': ['train.de.gz', 'train.en.gz'],
                                                   'valid': ['val.de.gz', 'val.en.gz'],
                                                   'test': ['test_2016_flickr.de.gz', 'test_2016_flickr.en.gz']}}

    tokenizer_training_params = {'vocab_size': 30000,
                                 'min_frequency': 2}

    training_params = {'hid_dim': 256,
                       'enc_layers': 3,
                       'dec_layers': 3,
                       'enc_heads': 8,
                       'dec_heads': 8,
                       'enc_pf_dim': 512,
                       'dec_pf_dim': 512,
                       'enc_dropout': 0.1,
                       'dec_dropout': 0.1,
                       'learning_rate': 0.0005,
                       'clip_grad': 1.,
                       'train_batch_size': 128,
                       'valid_batch_size': 128,
                       'num_epochs': 10}

    evaluation_params = {'test_batch_size': 128}

    # create all task specs
    dataset_loading_task = TaskSpec(task=DatasetLoading, task_kwargs=dataset_loading_params,
                                    publishes=['train_data', 'valid_data', 'test_data'])
    tokenizer_training_task = GridTaskSpec(task=TokenizerTraining, gs_config=tokenizer_training_params,
                                           expects=['train_data'],
                                           publishes=['de_tokenizer', 'en_tokenizer'])
    dataset_encoding_task = TaskSpec(task=DatasetEncoding,
                                     expects=['train_data', 'valid_data', 'test_data', 'de_tokenizer', 'en_tokenizer'],
                                     publishes=['train_encoded', 'valid_encoded', 'test_encoded'])
    train_task = GridTaskSpec(task=Training, gs_config=training_params,
                              expects=['train_encoded', 'valid_encoded', 'de_tokenizer', 'en_tokenizer'],
                              publishes=['best_model', 'best_model_metric'])
    model_selection_task = TaskSpec(task=ModelSelection, reduce=True, expects=['best_model_metric'],
                                    publishes=['best_run_config'])
    evaluate_task = TaskSpec(task=Evaluation, reduce=False, task_kwargs=evaluation_params,
                             expects=['best_run_config'], publishes=[])

    # dependencies between tasks
    tokenizer_training_task.requires([dataset_loading_task])
    dataset_encoding_task.requires([tokenizer_training_task, dataset_loading_task])
    train_task.requires([dataset_encoding_task, tokenizer_training_task])
    model_selection_task.requires([train_task])
    evaluate_task.requires([model_selection_task])

    # all tasks
    tasks = [dataset_loading_task, tokenizer_training_task, dataset_encoding_task,
             train_task, model_selection_task, evaluate_task]

    # create list of resources
    devices = get_balanced_devices(count=num_workers, use_cuda=use_cuda)
    resources = [TaskResource(device=devices[i], seed=seed) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=base_dir)

    with Swarm(n_dolphins=num_workers,
               resources=resources,
               results_store=results_store) as swarm:
        flow = Flow(swarm=swarm, task_to_execute=task_to_execute, force=force)
        flow.run(tasks)


if __name__ == '__main__':
    main()
