#%%
############################################################################
# IMPORTS
############################################################################
import os
import numpy as np
import h5py
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import altair as alt
import pandas as pd
from IPython.display import display

############################################################################
# Plotting Utilities, Constants, Methods for W209 arXiv project
############################################################################

#---------------------------------------------------------------------------
## Plotting Palette
#
# Create a dict object containing U.C. Berkeley official school colors for plot palette 
# reference : https://brand.berkeley.edu/colors/
# secondary reference : https://alumni.berkeley.edu/brand/color-palette# CLass Initialization
#---------------------------------------------------------------------------

berkeley_palette = {'berkeley_blue'     : '#003262',
                    'california_gold'   : '#fdb515',
                    'founders_rock'     : '#3b7ea1',
                    'medalist'          : '#c4820e',
                    'bay_fog'           : '#ddd5c7',
                    'lawrence'          : '#00b0da',
                    'sather_gate'       : '#b9d3b6',
                    'pacific'           : '#46535e',
                    'soybean'           : '#859438',
                    'south_hall'        : '#6c3302',
                    'wellman_tile'      : '#D9661F',
                    'rose_garden'       : '#ee1f60',
                    'golden_gate'       : '#ed4e33',
                    'lap_lane'          : '#00a598',
                    'ion'               : '#cfdd45',
                    'stone_pine'        : '#584f29',
                    'grey'              : '#eeeeee',
                    'web_grey'          : '#888888',
                    # alum only colors
                    'metallic_gold'     : '#BC9B6A',
                    'california_purple' : '#5C3160'                   
                    }

############################################################################
# VIZ : Define a Berkeley inspired Altair theme
############################################################################

def cal_theme():
    font = "Lato"

    return {
        "config": {
            "title": {
                "fontSize": 20,
                "font": font,
                "anchor": "middle",
                "align":"center",
                "color": berkeley_palette['berkeley_blue'],
                "subtitleFontSize": 15,
                "subtitleFont": font,
                "subtitleAcchor": "middle",
                "subtitleAlign": "center",
                "subtitleColor": berkeley_palette['pacific']
            },
            "axisX": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "axisY": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end",
                "gridColor": berkeley_palette['grey'],
                "gridWidth": 1
            },
            "headerRow": {
                "labelFont": font,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['berkeley_blue'],
                "titleAlign": "right",
                "titleAnchor": "end"
            },
            "legend": {
                "labelFont": font,
                "labelFontSize": 10,
                "labelColor": berkeley_palette['stone_pine'],
                "symbolType": "circle",
                "symbolSize": 120,
                "titleFont": font,
                "titleFontSize": 15,
                "titleColor": berkeley_palette['pacific']
            }
        }
    }

alt.themes.register("my_cal_theme", cal_theme)
alt.themes.enable("my_cal_theme")

#%%
############################################################################
# CLASS : BERTImageGenerator

#Example for obtaining data labels
#train = h5py.File('../SQuADv2/train_386.h5', 'r')
#start_ids = train['input_start']
#end_ids = train['input_end']
#labels = np.vstack([start_ids, end_ids]).T
############################################################################

class BERTImageGenerator(Sequence):
    """Keras data generator for reading .h5 files storing BERT images

    Args:
        data_dir ([str]): data directory containing images to load
        labels ([list]): ground-truth labels for the images
        batch_size ([int]): batch size of images to be returned 
        start_idx ([int]): starting index of the image generator
        end_idx ([int]): ending index of the image generator
        encoder_size ([int, optional]): encoder size for the BERT model, including output (13 for base, 24 for large)
        max_seq_length ([int, optional]): maximum query length for the model
        bert_embedding_dim ([int, optional]): BERT embedding dimension (786 for base, 1024 for large)
        include_output_seq ([bool, optional]): whether to include the output sequence in the training set
        shuffle ([bool, optional]): whether to shuffle data during training after each epoch
    Raises:
        RuntimeError: if data directory provided does not exist.
    """
    def __init__(self, 
                 data_dir,
                 labels, 
                 batch_size=32, 
                 start_idx = 0, 
                 end_idx = None, 
                 encoder_size = 25,
                 max_seq_length = 386,
                 bert_embedding_dim = 1024,
                 include_output_seq = False,
                 shuffle=True):
        
        if not os.path.isdir(data_dir):
            raise RuntimeError("Provided data directory does not exist")
            
        if end_idx is None:
            end_idx = len(labels)      
        
        self.shuffle = shuffle
        #keep track of data indices for generation
        self.indices = np.arange(start_idx, end_idx)
        self.labels = labels
        self.on_epoch_end()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.encoder_size = encoder_size
        self.max_seq_length = max_seq_length
        self.bert_embedding_dim = bert_embedding_dim
        self.include_output_seq = include_output_seq
        print('reloaded BERTImageGenerator')
        
    def __len__(self):
        '''Determines the number of batches per epoch'''
        return int(np.ceil((self.end_idx - self.start_idx) / self.batch_size))
    
    def __getitem__(self, idx):
        '''Retrieves the batch of examples'''
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        return self.__data_generator(indices)
        
    def __data_generator(self, indices):
        '''Generates a batch of data based on the sequences'''
        encoder_size = self.encoder_size
        if not self.include_output_seq:
            encoder_size -= 1
            
        X = np.empty((len(indices), encoder_size, self.max_seq_length, self.bert_embedding_dim))
        
        for i,idx in enumerate(indices):
            with h5py.File(self.data_dir + '/' + str(idx) + '.h5', 'r') as f_in:
                
                embedding = pad_sequences(f_in['hidden_state_activations'], 
                                          self.max_seq_length, 
                                          dtype = np.float32)
                
                if self.include_output_seq:
                    X[i] = embedding
                else:
                    X[i] = embedding[:-1]
        
        labels = self.labels[indices].T
        return X, [labels[0], labels[1]]
        #return np.swapaxes(X, 1, 3), [labels[0], labels[1]]
    
    def on_epoch_end(self):
        '''Performs data shuffling at the end of each epoch'''
        if self.shuffle == True:
            idx_shuffle = np.arange(len(self.indices), dtype = int)
            np.random.shuffle(idx_shuffle)
            self.indices = self.indices[idx_shuffle]

def simple_generator(labels, directory, batch_size = 32, truncate_data = False, shuffle_index = None):
    
    indices = np.arange(len(labels), dtype = int)
    
    if shuffle_index is None:
        np.random.shuffle(indices)
    else:
        indices = indices[shuffle_index]
    
    if truncate_data:
        indices = indices[:truncate_data]
        
    darray = np.zeros((batch_size,386,1024,3), dtype = np.float32)
    
    offset = 0
    while True:
        darray[:] = 0
        data_slice = indices[offset * batch_size:(offset+1)*batch_size]
        for i,idx in enumerate(data_slice):
            with h5py.File(directory + '%d.h5' %(idx), 'r') as f:
                data = np.array(f['hidden_state_activations'], dtype = np.float32)
            darray[i, 386-data.shape[0]:386, :, :] = data
        #print(darray[0][-1][-1][-1])
        if len(data_slice) < batch_size:
            output = darray[:len(data_slice)]
            offset = 0
        else:
            offset += 1
            output = darray
        
        #print(output.shape, labels[data_slice].shape)
        yield output, list(labels[data_slice].T)
        
def simple_dev_generator(labels, directory, batch_size = 32, truncate_data = False):
    indices = np.arange(len(labels), dtype = int)
    #np.random.shuffle(indices)
        
    darray = np.zeros((batch_size,386,1024,3), dtype = np.float32)
    
    offset = 0
    while True:
        data_slice = indices[offset * batch_size:(offset+1)*batch_size]
        for i,idx in enumerate(data_slice):
            with h5py.File(directory + '%d.h5' %(idx), 'r') as f:
                data = np.array(f['hidden_state_activations'], dtype = np.float32)
            darray[i, 386-data.shape[0]:386] = data
        print(offset)
        if len(data_slice) < batch_size:
            output = darray[:len(data_slice)]
            offset = 0
        else:
            offset += 1
            output = darray
        yield output

#%%
############################################################################
# VIZ Utilities
#   Visualization utiltiies for training
############################################################################

def __validate_perf_df(df, expected_cols = ['epoch', 'f1', 'em']):
    """Validates the input dataframe for type, shape, and column

    Args:
        df (Pandas.DataFrame): Dataframe to validation for input
        expected_cols (list, optional): Which column names to verify exist in the dataframe. Defaults to ['epoch', 'f1', 'em'].

    Raises:
        ValueError: if `df` parameter is not of type Pandas.DataFrame
        ValueError: if `df` dataframe is empty
        ValueError: if `df` does not contain the `expected_cols` columns.
    """

    ### Input Validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input parameter `df` must be of type Pandas.DataFrame. Provided value was of type '{type(df)}'.'")
    if df.shape[0] < 1:
        raise ValueError("Input parameter `df` must contain at least one row of data.")
    for c in expected_cols:
        if not c in df.columns:
            raise ValueError(f"Input paramter `df` must contain column for '{c}' values.")

    return

def viz_table_scores(df, title = 'BERT-large Binary Classification on SQuAD v2', cmap = 'BuGn'):
    """Produces a styled Pandas table output of F1 & EM performance metrics for BERTVision tasks.

    Args:
        df (Pandas.DataFrame): Pandas dataframe of the F1 and EM (exact match) metrics of of fine-tuning BERT for SQuAD v2 tasks.
            Columns expected:
            * epoch (float16) : ex. 0.1, 0.2, 0.3, ..., 5, 6, 7, ...
            * f1 (float16) : ex. 0.75644543, 0.6560445, ...
            * em (float16) : ex. 0.7843433, 0.832325, ...
        title (str): title of the table (caption)
        cmap (str): color map defined for pandas styler
    """

    # perform input validation
    expected_cols = ['epoch', 'f1', 'em']
    __validate_perf_df(df = df, expected_cols = expected_cols)


    # limit the input dataframe to only those columns we need, and sorted by epoch
    df = df[expected_cols].sort_values(by = 'epoch', ascending = True)

    format_dict = {'epoch':'{:.1f}', 'f1': '{:.3%}', 'em': '{:.3%}'}

    out = df[(df.epoch > 0)].style.set_caption(title)\
        .format(format_dict)\
        .background_gradient(subset=['f1', 'em'], cmap = cmap).hide_index()

    display(out)

    return

def viz_bert_performance(df, xaxis_title = 'Training Epochs', yaxis_title = 'Performance Metric (Dev)',
    legend_title = 'Metric', title = 'BERT-large Binary Classification Fine-Tuned Performance',
    subtitle = 'SQuAD v2 (Answer / No Answer Detection', width = 800, height = 400):
    """Returns an Altair visual of BERT fine-tune performance F1 / EM metrics

    Args:
        df (Pandas.DataFrame): Pandas dataframe of the F1 and EM (exact match) metrics of of fine-tuning BERT for SQuAD v2 tasks.
            Columns expected:
            * epoch (float16) : ex. 0.1, 0.2, 0.3, ..., 5, 6, 7, ...
            * f1 (float16) : ex. 0.75644543, 0.6560445, ...
            * em (float16) : ex. 0.7843433, 0.832325, ...
        xaxis_title (str, optional): Label for the X axis. Defaults to 'Training Epochs'.
        yaxis_title (str, optional): Label for the Y axis. Defaults to 'Performance Metric (Dev)'.
        legend_title (str, optional): Label for the Legend. Defaults to 'Metric'.
        title (str, optional): Title of the report. Defaults to 'BERT-large Binary Classification Fine-Tuned Performance'.
        subtitle (str, optional): Subtitle of the report. Defaults to 'SQuAD v2 (Answer / No Answer Detection'.
        width (int, optional): Width of the visual in pixels. Defaults to 800.
        height (int, optional): Height of the visual in pixels. Defaults to 400.
    """

    # perform input validation
    expected_cols = ['epoch', 'f1', 'em']
    __validate_perf_df(df = df, expected_cols = expected_cols)

    # limit the input dataframe to only those columns we need, and sorted by epoch
    df = df[expected_cols].sort_values(by = 'epoch', ascending = True)

    # transform the dataframe for visualization
    adf = pd.melt(df, id_vars = ['epoch'], value_vars = ['f1','em'], value_name = 'score', var_name = 'metric')

    min_epoch, max_epoch = min(adf.epoch), max(adf.epoch)
    min_score, max_score = min(adf.score), max(adf.score)
    xaxis_bins = df.epoch.values.astype(np.float16)
    metric_range = [berkeley_palette['berkeley_blue'], berkeley_palette['lawrence']]

    c = alt.Chart(adf).mark_line().encode(
        x = alt.X('epoch:Q', axis = alt.Axis(title = xaxis_title, grid = False, 
                formatType = "number", format = ".1", labelOverlap = True),
            scale = alt.Scale(domain = [min_epoch - 1, max_epoch], bins = xaxis_bins)),
        y = alt.Y('score:Q', axis = alt.Axis(title = yaxis_title, grid = True),
            scale = alt.Scale(domain = [min_score - (.05 * min_score), max_score + (.05 * max_score)])),
        color = alt.Color('metric',
            scale = alt.Scale(domain = ['f1','em'], range = metric_range),
            legend = alt.Legend(title = legend_title))
        ).properties(width = width, height = height).configure_view(strokeWidth = 0)

    display(c.properties(title = {"text": title, "subtitle": subtitle}))

    return


# %%
