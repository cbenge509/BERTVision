# packages
import time, os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from utils.collate import collate_squad_train, collate_squad_dev, collate_squad_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from elasticsearch import Elasticsearch
from datasets import load_dataset, load_metric
from utils.squad_preprocess import prepare_validation_features, postprocess_qa_predictions
import subprocess
subprocess.Popen('C:\\elasticsearch-7.10.2\\bin\\elasticsearch.bat')


class BertQAEvaluator(object):
    '''
    This class handles the evaluation of QA models with BERT architecture.

    Parameters
    ----------
    model : object
        A HuggingFace QuestionAnswering BERT transformer

    processor : object
        A Torch Dataset processor that emits data

    args : object
        An Argument Parser object; see args.py

    Returns
    ----------
    metrics : float
        QA-specified metrics; e.g., F1 and EM

    '''
    def __init__(self, model, processor, args):
        # pull in init
        self.args = args
        self.model = model
        self.processor = processor

        # set placeholders for evaluation metrics
        self.dev_loss = 0
        self.nb_dev_steps = 0
        self.nb_dev_examples = 0

    def get_loss(self):
        '''
        This function prepares the data and handles the validation set testing.
        '''
        # instantiate dev set processor
        self.dev_examples = self.processor(type='dev')
        # craete dev set data loader
        dev_dataloader = DataLoader(self.dev_examples,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    drop_last=False,
                                    collate_fn=collate_squad_dev)
        # set the model to evaluation
        self.model.eval()
        # for each batch of data,
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            # send it to the GPU
            input_ids, attn_mask, start_pos, end_pos, token_type_ids = (
                batch['input_ids'].to(self.args.device),
                batch['attention_mask'].to(self.args.device),
                batch['start_positions'].to(self.args.device),
                batch['end_positions'].to(self.args.device),
                batch['token_type_ids'].to(self.args.device),
            )

            # forward
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                    token_type_ids=token_type_ids)

            # loss
            if self.args.n_gpu > 1:
                loss = out.loss.mean()
            else:
                loss = out.loss

            # metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += input_ids.size(0)
            self.nb_dev_steps += 1
        # calculate end of training loss
        avg_loss = self.dev_loss / self.nb_dev_steps
        # report it and return it
        print('\n', 'dev loss', self.dev_loss / self.nb_dev_steps)
        return avg_loss

    def get_scores(self):
        '''
        This function prepares the data to generate start and end logit preds.
        '''
        # instantiate dev set processor for predictions
        self.score_examples = self.processor(type='score')
        # prepare a data loader
        scores_dataloader = DataLoader(self.score_examples,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      drop_last=False,
                                      collate_fn=collate_squad_score)
        # ensure model is still on evaluation
        self.model.eval()
        # create containers for logits and indices
        start_logits = []
        end_logits = []
        indices = []
        # for each batch
        for step, batch in enumerate(tqdm(scores_dataloader, desc="Scoring")):
            # send them to the GPU
            input_ids, attn_mask, token_type_ids, idx = (
                batch['input_ids'].to(self.args.device),
                batch['attention_mask'].to(self.args.device),
                batch['token_type_ids'].to(self.args.device),
                batch['idx'].to(self.args.device)
                )
            # forward
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids
                    )

            # store the logits and indices for retrieval
            start_logits.append(out.start_logits.detach().cpu().numpy())
            end_logits.append(out.end_logits.detach().cpu().numpy())
            indices.append(idx.detach().cpu().numpy())
        # concatenate them together and return them
        indices = np.concatenate(indices)
        logits = {'start_logits': np.concatenate(start_logits),
                  'end_logits': np.concatenate(end_logits)}
        return logits, indices

    # score val squad
    def score_squad_val(self, shuffled_idx, logits, n_best_size, max_answer):
        '''
        This function incorporates HuggingFace's official scoring function which
        scores all the examples at the end of training. Prior to using their function,
        it is necessary to notice several things and take action against them.
            (1) Our torch data set and data loaders shuffle the data
                considerably. As such, we need track the shuffle of the training
                data and then use this information to re-order the original
                SQuAD data so the first sample emitted is scored against its
                identical location in the original SQuAD data. To do so,
                we use select to order the indices by the training order and then
                use the HuggingFace's data set class ElasticSearch feature to
                find each example_id. Making things more complicated, some
                SQuAD samples contain so much text that they are broken up many
                times, which is why SQuAD2's 130,319 training examples are
                transformed into 131,754 training examples. This is why we work
                with multiple data sets labelled examples and features. Once
                examples and features are re-ordered by our training, we can
                finally compute our metrics.
        '''
        # build base examples, features set of training data
        examples = load_dataset("squad_v2").shuffle(seed=1)['validation']
        features = load_dataset("squad_v2").shuffle(seed=1)['validation'].map(
            prepare_validation_features,
            batched=True,
            remove_columns=['answers', 'context', 'id', 'question', 'title'])
        # reorder features by the training process
        features = features.select(indices=shuffled_idx)
        # get the example ids to match with the "example" data; get unique entries
        id_list = list(dict.fromkeys(features['example_id']))
        # now search for their index positions; load elastic search
        es = Elasticsearch([{'host': 'localhost'}]).ping()
        # add an index to the id column for the examples
        examples.add_elasticsearch_index(column='id')
        # give it time to process
        time.sleep(30)  # 30 seconds
        # retrieve the example index
        example_idx = [examples.search(index_name='id', query=i, k=1).indices for i in id_list]
        # flatten
        example_idx = [item for sublist in example_idx for item in sublist]
        # drop the index
        examples.drop_index(index_name='id')
        # put examples in the right order
        examples = examples.select(indices=example_idx)
        # flatten indicies -- speeds up process significantly
        examples = examples.flatten_indices()
        features = features.flatten_indices()
        # proceed with QA calculation
        final_predictions = postprocess_qa_predictions(examples=examples,
                                                       features=features,
                                                       starting_logits=logits['start_logits'],
                                                       ending_logits=logits['end_logits'],
                                                       n_best_size=n_best_size,
                                                       max_answer_length=max_answer)
        # load HF metric for squad2
        metric = load_metric("squad_v2")
        # reformat the preds
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
        # format the references
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        # compute the final metrics
        metrics = metric.compute(predictions=formatted_predictions, references=references)
        return metrics

#
