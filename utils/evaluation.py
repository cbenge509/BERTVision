#%%
############################################################################
# IMPORTS
############################################################################

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import json
import h5py
import pickle
import os
from collections import defaultdict, Counter


from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from transformers.data.processors.squad import SquadV2Processor
from transformers.data.processors.squad import squad_convert_examples_to_features

#%%
############################################################################
# CLASS : Squad2Config
# TODO: Merge with Cris's utils
# TODO: Make class a global configs utility
# TODO: Move model elsewhere
############################################################################

class Squad2Config(object):
    """Stores configurations for metric evaluation to ensure consistency

    Args:
        named_model ([str]): huggingface model name (for both BertLayer and Tokenizer)
        max_query_length ([int]): maximum length of question before truncation
        max_seq_length ([int]): maximum length of context after question and before truncation
        doc_stride ([int]): overlap between context splits when context exceeds max_seq_length
        processor ([int]): Squad data processor from huggingface

    """

    def __init__(self,
                 named_model = 'bert-large-uncased',
                 max_query_length = 64,
                 max_seq_length = 386,
                 doc_stride = 128,
                 processor = SquadV2Processor(),
                 ):

        self.named_model = named_model
        self.tokenizer = BertTokenizer.from_pretrained(named_model)
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.processor = processor

class SquadAnswers(object):
    """Base class for processing Squad possible answers

    Args:
        examples ([list]): list of huggingface processed examples
        embeddings ([array]): OS path location of the output folder where the h5 processed SQuAD data should be stored.
        is_training ([bool]): True for training data; False for dev
        qas_ids ([multi]): (Optional) file for pre-ordered question ids. If unspecified, will extract from raw json
        config ([Squad2Config]): Squad configuration
    Raises:
        TypeError: if "examples" does not contain qas_id and answers attributes
        RuntimeError: if embeddings path does not exist.
    """

    def __init__(self,
                 examples,
                 embeddings,
                 is_training,
                 qas_ids = None,
                 config = Squad2Config()):

        self.config = config
        self.tokenizer = config.tokenizer

        if not isinstance(examples, list) or not hasattr(examples[0], 'answers'):
            raise TypeError ("examples must come from a huggingface Squad processor")

        self.examples = examples
        self.is_training = is_training

        self.qas_ids = self.get_qas_ids(qas_ids)

        #Initialize data
        if not os.path.isfile(embeddings):
            raise RuntimeError("Provided embeddings file does not exist. Please specify with correct path, or generate using utils.")

        data = h5py.File(embeddings, 'r')
        self.ids = np.array(data['input_ids'], dtype = np.int32)
        self.masks = np.array(data['attention_mask'], dtype = np.int32)

    def get_qas_ids(self, qas_ids):
        if os.path.isfile(qas_ids):
            qas_ids = pickle.load(open(qas_ids, 'rb'))
        else:
            dataset = squad_convert_examples_to_features(
                examples=self.examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
                doc_stride=self.config.doc_stride,
                max_query_length=self.config.max_query_length,
                is_training=self.is_training,
            )
            qas_ids = dataset.qas_id
        return qas_ids

    def get_all_possible_answers(self):
        '''Squad examples often provide multiple correct answers. This function collates all answers based on qas_id'''
        answers_based_on_id = defaultdict(lambda: [])

        for example in self.examples:

            #examples must store a set of ground-true qas_ids
            #which will be used to match current id
            qas_id = example.qas_id
            answers = example.answers

            if answers:
                for answer in answers:
                    tokenized_answer = self.tokenizer.tokenize(answer['text'])
                    answers_based_on_id[qas_id].append(tokenized_answer)
            else:
                answers_based_on_id[qas_id] = []
        return answers_based_on_id

class SquadDevAnswers(SquadAnswers):
    """Subclass for processing Squad dev answers

    Args:
        data_dir ([dir]): directory containg squad dev data
        embeddings ([array]): see SquadAnswers
        qas_ids ([multi]): see SquadAnswers
        config ([Squad2Config]): see SquadAnswers
    Raises:
        RuntimeError: if data_dir does not exist.
    """

    def __init__(self,
                 data_dir,
                 dev_embeddings,
                 qas_ids = None,
                 config = Squad2Config()):

        if not os.path.isdir(data_dir):
            raise RuntimeError("data_dir path not found")

        self.dev_examples = config.processor.get_dev_examples(data_dir)

        super().__init__(self.dev_examples,
                         dev_embeddings,
                         is_training = False,
                         qas_ids = qas_ids,
                         config = config
                         )

class SquadModelEvaluation(object):
    """Subclass for processing Squad dev answers

    Args:
        model ([keras_model]): model with properly loaded weights for evaluation against the dev set
        squad_answers ([SquadAnswers]): see SquadAnswers
        start_logits ([array]): (Optional) logits from model for the start span
        end_logits ([array]): (Optional) logits from model for the end span
    """

    def __init__(self,
                 model,
                 squad_answers,
                 start_logits = None,
                 end_logits = None,
                 already_prob = False):

        self.squad_answers = squad_answers

        #Initialize model predictions
        if start_logits is None or end_logits is None:
            start_logits, end_logits = model.predict([self.squad_answers.ids, self.squad_answers.masks])

        self.start_logits, self.end_logits = start_logits, end_logits

        self.already_prob = already_prob
        
        self.true_answers = squad_answers.get_all_possible_answers()
        self.predicted_answers = self.get_best_prob_answer()

    def get_output_strings(self):
        '''Return full tokens for all predictions'''
        start_tokens = self.start_logits.argmax(axis = 1)
        end_tokens = self.end_logits.argmax(axis = 1)

        answers = []

        for i,example in enumerate(self.squad_answers.ids):
            start_ind = start_tokens[i]
            end_ind = end_tokens[i] + 1
            answer = self.squad_answers.tokenizer.convert_ids_to_tokens(example[start_ind:end_ind])
            if answer == ['[CLS]']:
                answer = []
            answers.append(answer)
        return answers

    def get_per_example_probs(self):
        '''Return probabilities for each example'''

        if not self.already_prob:
            start_softmax = np.exp(self.start_logits)/np.sum(np.exp(self.start_logits), axis = 1, keepdims = True)
            end_softmax = np.exp(self.end_logits)/np.sum(np.exp(self.end_logits), axis = 1, keepdims = True)
            logits_start = start_softmax.max(axis = 1)
            logits_end = end_softmax.max(axis = 1)
        else:
            logits_start = self.start_logits.max(axis = 1)
            logits_end = self.end_logits.max(axis = 1)
            
        prob = logits_start * logits_end
            
        return prob

    def get_best_prob_answer(self):
        '''
        For dev set, examples exceeding 386 tokens are split into multiple documents with doc stride (overlapping) text
            of 128 tokens. Predictions are made on each of these splits, and this function returns best answer based on
            normalized probability of the predictions
        '''

        best_answers = {}

        answers = self.get_output_strings()
        probs = self.get_per_example_probs()
        
        for i, answer in enumerate(answers):
            prob = probs[i]
            qas_id = self.squad_answers.qas_ids[i]

            if qas_id not in best_answers:
                best_answers[qas_id] = [answer, prob]
            else:
                if prob > best_answers[qas_id][1]:
                    best_answers[qas_id] = [answer, prob]

        return best_answers

    def evaluate_metrics(self,
                         metric = 'all'):
        '''Evaluates Squad metrics: Exact Match and F1 based on token overlap'''
        EM = []
        F1 = []

        answers = self.predicted_answers
        true_answers = self.true_answers

        for i, qas_id in enumerate(answers):
            answer = answers[qas_id][0]
            true_answer = true_answers[qas_id]

            em = f1 = 0
            if true_answer == []:
                if answer == []:
                    em = f1 = 1
            else:
                f1_scores = []
                for possible_answer in true_answer:
                    if answer == possible_answer:
                        em = 1

                    common_tokens = Counter(answer) & Counter(possible_answer)
                    number_same = sum(common_tokens.values())

                    if number_same == 0:
                        f1_scores.append(0)
                    else:
                        precision = number_same / len(answer)
                        recall = number_same / len(possible_answer)
                        f1_scores.append((2 * precision * recall)/(precision + recall))

                f1 = max(f1_scores)

            EM.append(em)
            F1.append(f1)

        print("Exact match: %.3f" %(sum(EM)/len(EM)) + '\n' + "F1 score: %.3f" %np.mean(F1))


    def eval_F1(self):

        all_answers = []
        answers = self.predicted_answers
        true_answers = self.true_answers

        for i, qas_id in enumerate(answers):
            answer = answers[qas_id][0]
            true_answer = true_answers[qas_id]

            score = 0
            if true_answer == []:
                if answer == []:
                    score = 1
            else:
                f1_scores = []
                for possible_answer in true_answer:
                    common_tokens = Counter(answer) & Counter(possible_answer)
                    number_same = sum(common_tokens.values())

                    if number_same == 0:
                        f1_scores.append(0)
                    else:
                        precision = number_same / len(possible_answer)
                        recall = number_same / len(answer)
                        f1_scores.append((2 * precision * recall)/(precision + recall))
                score = max(f1_scores)
            all_answers.append(score)

        return np.mean(all_answers)

class PretrainedBertSquad2(object):
    def __init__(self,
                 weights,
                 config = Squad2Config()):

        self.tokenizer = config.tokenizer
        self.named_model = config.named_model
        self.model = self.bert_large_uncased_for_squad2(config.max_seq_length)
        self.model.load_weights(weights)

    def bert_large_uncased_for_squad2(self, max_seq_length):
        input_ids = Input((max_seq_length,), dtype = tf.int32, name = 'input_ids')
        input_masks = Input((max_seq_length,), dtype = tf.int32, name = 'input_masks')

        #Load model from huggingface
        bert_layer = TFBertModel.from_pretrained(self.named_model)

        outputs = bert_layer([input_ids, input_masks])[0] #1 for pooled outputs, 0 for sequence

        #Dense layer with 2 nodes; one for start span and one for end span
        logits = Dense(2)(outputs)

        #Split the outputs into start and end logits
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = K.squeeze(start_logits, axis=-1)
        end_logits = K.squeeze(end_logits, axis=-1)

        model = Model(inputs = [input_ids, input_masks], outputs = [start_logits, end_logits])
        return model

class PretrainedBertSquad2Faster(object):
    def __init__(self,
                 weights,
                 config = Squad2Config()):

        self.config = config
        self.tokenizer = config.tokenizer
        self.named_model = config.named_model
        self.model = self.bert_large_uncased_for_squad2(self.config.max_seq_length)
        self.model.load_weights(weights)

    def bert_large_uncased_for_squad2(self, max_seq_length):
        input_ids = Input((max_seq_length,), dtype = tf.int32, name = 'input_ids')
        input_masks = Input((max_seq_length,), dtype = tf.int32, name = 'input_masks')

        #Load model from huggingface
        config = BertConfig.from_pretrained(self.config.named_model, output_hidden_states=True)
        bert_layer = TFBertModel.from_pretrained(self.named_model, config = config)

        outputs, _, embeddings = bert_layer([input_ids, input_masks]) #1 for pooled outputs, 0 for sequence

        model = Model(inputs = [input_ids, input_masks], outputs = [embeddings, outputs])
        return model
