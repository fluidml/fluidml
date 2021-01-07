from dataclasses import dataclass
from datetime import datetime
import math
import multiprocessing
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import spacy
import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score

from transformer_model import Seq2SeqTransformer, Encoder, Decoder
from fluidml.common import Task, Resource
from fluidml.swarm import Swarm
from fluidml.flow import Flow, GridTaskSpec, TaskSpec


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
    torch.backends.cudnn.deterministic = True


@dataclass
class TaskResource(Resource):
    device: str
    seed: int


class DatasetPreparation(Task):
    def __init__(self,
                 batch_size: int = 128,
                 min_freq: int = 2):
        super().__init__()

        self.batch_size = batch_size
        self.min_freq = min_freq

        self.spacy_de, self.spacy_en = self.load_spacy_models()

    @staticmethod
    def load_spacy_models():
        try:
            spacy_de = spacy.load('de')
        except OSError:
            spacy.cli.download('de')
            spacy_de = spacy.load('de')

        try:
            spacy_en = spacy.load('en')
        except OSError:
            spacy.cli.download('en')
            spacy_en = spacy.load('en')
        return spacy_de, spacy_en

    def tokenize_de(self, text: str) -> List:
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> List:
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def run(self):
        set_seed(self.resource.seed)

        source_field = Field(tokenize=self.tokenize_de,
                             init_token='<sos>',
                             eos_token='<eos>',
                             lower=True,
                             batch_first=True)

        target_field = Field(tokenize=self.tokenize_en,
                             init_token='<sos>',
                             eos_token='<eos>',
                             lower=True,
                             batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(source_field, target_field))

        source_field.build_vocab(train_data, min_freq=self.min_freq)
        target_field.build_vocab(train_data, min_freq=self.min_freq)

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.batch_size,
            device=self.resource.device)

        return {'train_iterator': train_iterator,
                'valid_iterator': valid_iterator,
                'test_iterator': test_iterator,
                'test_data': test_data,
                'source_field': source_field,
                'target_field': target_field}


class Training(Task):
    def __init__(self,
                 hid_dim: int = 256,
                 enc_layers: int = 3,
                 dec_layers: int = 3,
                 enc_heads: int = 8,
                 dec_heads: int = 8,
                 enc_pf_dim: int = 512,
                 dec_pf_dim: int = 512,
                 enc_dropout: float = 0.1,
                 dec_dropout: float = 0.1,
                 learning_rate: float = 0.0005,
                 clip_grad: float = 1.,
                 num_epochs: int = 10):
        super().__init__()

        self.hid_dim = hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        self.enc_pf_dim = enc_pf_dim
        self.dec_pf_dim = dec_pf_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.num_epochs = num_epochs

    def _init_training(self, input_dim: int, output_dim: int, src_pad_idx: int, trg_pad_idx: int, device: torch.device):
        enc = Encoder(input_dim,
                      self.hid_dim,
                      self.enc_layers,
                      self.enc_heads,
                      self.enc_pf_dim,
                      self.enc_dropout,
                      device)

        dec = Decoder(output_dim,
                      self.hid_dim,
                      self.dec_layers,
                      self.dec_heads,
                      self.dec_pf_dim,
                      self.dec_dropout,
                      device)

        model = Seq2SeqTransformer(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
        return model, optimizer, criterion

    def _train_epoch(self, model, iterator, optimizer, criterion):
        model.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

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
        model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for batch in iterator:
                src = batch.src
                trg = batch.trg

                output, _ = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)  # [batch size * trg len - 1, output dim]
                trg = trg[:, 1:].contiguous().view(-1)             # [batch size * trg len - 1]

                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def _train(self, model, train_iterator, valid_iterator, optimizer, criterion):
        best_valid_loss = float('inf')
        best_model = None

        for epoch in range(self.num_epochs):

            start_time = datetime.now()
            train_loss = self._train_epoch(model, train_iterator, optimizer, criterion)
            valid_loss = self.validate_epoch(model, valid_iterator, criterion)
            end_time = datetime.now()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = model.state_dict()
                # torch.save(model.state_dict(), 'best_model.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {end_time - start_time}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        assert best_model is not None
        return best_model, best_valid_loss

    def run(self, source_field, target_field, train_iterator, valid_iterator, test_iterator):
        set_seed(self.resource.seed)

        input_dim = len(source_field.vocab)
        output_dim = len(target_field.vocab)
        src_pad_idx = source_field.vocab.stoi[source_field.pad_token]
        trg_pad_idx = target_field.vocab.stoi[target_field.pad_token]

        model, optimizer, criterion = self._init_training(input_dim=input_dim,
                                                          output_dim=output_dim,
                                                          src_pad_idx=src_pad_idx,
                                                          trg_pad_idx=trg_pad_idx,
                                                          device=self.resource.device)

        best_model, best_valid_loss = self._train(model=model,
                                                  train_iterator=train_iterator,
                                                  valid_iterator=valid_iterator,
                                                  optimizer=optimizer,
                                                  criterion=criterion)
        return {'best_model': best_model,
                'best_valid_loss': best_valid_loss}


class Evaluation(Task):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _unpack_results(results: Dict[str, Any]) -> Tuple:
        training_results = results['Training']
        dataset_prep_results = results['DatasetPreparation']['result']
        source_field = dataset_prep_results['source_field']
        target_field = dataset_prep_results['target_field']
        test_iterator = dataset_prep_results['test_iterator']
        test_data = dataset_prep_results['test_data']
        return training_results, source_field, target_field, test_iterator, test_data

    @staticmethod
    def _init_model(train_config: Dict, device: torch.device,
                    input_dim: int, output_dim: int, src_pad_idx: int, trg_pad_idx: int) -> nn.Module:
        enc = Encoder(input_dim,
                      train_config['hid_dim'],
                      train_config['enc_layers'],
                      train_config['enc_heads'],
                      train_config['enc_pf_dim'],
                      train_config['enc_dropout'],
                      device)

        dec = Decoder(output_dim,
                      train_config['hid_dim'],
                      train_config['dec_layers'],
                      train_config['dec_heads'],
                      train_config['dec_pf_dim'],
                      train_config['dec_dropout'],
                      device)

        model = Seq2SeqTransformer(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)
        return model

    @staticmethod
    def _select_model(training_results: List[Dict]) -> Tuple:
        config = None
        best_model = None
        best_valid_loss = float('inf')
        for sweep in training_results:
            if sweep['result']['best_valid_loss'] < best_valid_loss:
                best_valid_loss = sweep['result']['best_valid_loss']
                best_model = sweep['result']['best_model']
                config = sweep['config']
        return best_model, config

    @staticmethod
    def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
        model.eval()

        if isinstance(sentence, str):
            nlp = spacy.load('de')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)

        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        return trg_tokens[1:], attention

    @staticmethod
    def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
        trgs = []
        pred_trgs = []

        for datum in data:
            src = vars(datum)['src']
            trg = vars(datum)['trg']

            pred_trg, _ = Evaluation.translate_sentence(src, src_field, trg_field, model, device, max_len)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

        return bleu_score(pred_trgs, trgs)

    def run(self, results, resource):
        device = resource.device
        seed = resource.seed
        set_seed(seed)

        training_results, source_field, target_field, test_iterator, test_data = self._unpack_results(results=results)

        input_dim = len(source_field.vocab)
        output_dim = len(target_field.vocab)
        src_pad_idx = source_field.vocab.stoi[source_field.pad_token]
        trg_pad_idx = target_field.vocab.stoi[target_field.pad_token]

        best_model, config = self._select_model(training_results=training_results)

        model = self._init_model(train_config=config['Training'],
                                 device=device,
                                 input_dim=input_dim,
                                 output_dim=output_dim,
                                 src_pad_idx=src_pad_idx,
                                 trg_pad_idx=src_pad_idx)
        model.load_state_dict(best_model)

        criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

        test_loss = Training.validate_epoch(model=model, iterator=test_iterator, criterion=criterion)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        bleu = self.calculate_bleu(test_data, source_field, target_field, model, device)
        print(f'BLEU score = {bleu * 100:.2f}')

        return {'best_model': best_model, 'config': config}


def main():
    base_dir = 'bla'
    num_workers = 4
    use_cuda = True
    seed = 1234

    dataset_preparation_params = {'batch_size': 128,
                                  'min_freq': 2}

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
                       'num_epochs': 10}

    # create all task specs
    dataset_prep_task = GridTaskSpec(task=DatasetPreparation, gs_config=dataset_preparation_params)
    train_task = GridTaskSpec(task=Training, gs_config=training_params)
    evaluate_task = TaskSpec(task=Evaluation, reduce=True)

    # dependencies between tasks
    train_task.requires([dataset_prep_task])
    evaluate_task.requires([train_task, dataset_prep_task])

    # all tasks
    tasks = [dataset_prep_task, train_task, evaluate_task]

    # create list of resources
    devices = get_balanced_devices(count=num_workers, use_cuda=use_cuda)
    resources = [TaskResource(device=devices[i], seed=seed) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = LocalFileStore(base_dir=base_dir)

    with Swarm(n_dolphins=num_workers,
               resources=resources,
               results_store=results_store) as swarm:
        flow = Flow(swarm=swarm)
        results = flow.run(tasks)


if __name__ == '__main__':
    main()
