# packages
import os, sys, datetime, time
sys.path.append("C:/BERTVision/code/torch")
from utils.collate import collate_H5_squad
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from elasticsearch import Elasticsearch
from datasets import load_dataset, load_metric
from utils.squad_preprocess import prepare_validation_features, postprocess_qa_predictions
import subprocess
subprocess.Popen('C:\\elasticsearch-7.10.2\\bin\\elasticsearch.bat')


class H5_SQUAD_Evaluator(object):
    '''
    This class handles the evaluation of 1-epoch tuned QA embeddings from BERT

    Parameters
    ----------
    model : object
        A huggingface QuestionAnswering BERT transformer

    criterion : loss function
        A loss function

    processor: object
        A Torch Dataset processor that emits data

    args: object
        A argument parser object; see args.py

    '''
    def __init__(self, model, criterion, processor, args):
        # pull in init
        self.args = args
        self.model = model
        self.criterion = criterion
        self.processor = processor

        # set placeholders for evaluation metrics
        self.dev_loss = 0
        self.nb_dev_steps = 0
        self.nb_dev_examples = 0

    def get_loss_and_scores(self):
        '''
        This function prepares the data and handles the validation set testing.
        '''
        # instantiate dev set processor
        self.dev_examples = self.processor(type='dev', args=self.args)
        # craete dev set data loader
        dev_dataloader = DataLoader(self.dev_examples,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    drop_last=False,
                                    collate_fn=collate_H5)
        # set the model to evaluation
        self.model.eval()

        # create containers for logits and indices
        start_logits = []
        end_logits = []
        indices = []

        # for each batch of data,
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            # send it to the GPU
            embeddings, start_ids, end_ids, idx = (
                batch['embeddings'].to(self.args.device),
                batch['start_ids'].to(self.args.device),
                batch['end_ids'].to(self.args.device),
                batch['idx'].to(self.args.device)
            )

            # forward
            with torch.no_grad():
                start, end = self.model(embeddings)

            # get loss for start and ending positions
            start_loss = self.criterion(start, start_ids)
            end_loss = self.criterion(end, end_ids)
            # combine them
            total_loss = (start_loss + end_loss) / 2

            # loss
            if self.args.n_gpu > 1:
                raise NotImplementedError
            else:
                loss = total_loss

            # we need to save all the logits for predictions; detach from GPU mem.
            start_logits.append(start.detach().cpu().numpy())
            end_logits.append(end.detach().cpu().numpy())
            indices.append(idx.cpu().numpy())

            # metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += embeddings.size(0)
            self.nb_dev_steps += 1

        # calculate end of training loss
        avg_loss = self.dev_loss / self.nb_dev_steps
        # report it and return it
        print('\n', 'dev loss', avg_loss)

        # concatenate indices and logits together and return them
        indices = np.concatenate(indices)
        logits = {'start_logits': np.concatenate(start_logits),
                  'end_logits': np.concatenate(end_logits)}
        return avg_loss, logits, indices

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
