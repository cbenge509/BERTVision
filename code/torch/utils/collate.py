import torch

# collate fn for HuggingFace data
def collate_squad_train(batch):
    '''
    This function collates the data emitted from the torch data set class
    and emits it as a dictionary.

    Parameters
    ----------
    batch : torch tensors
        The data loader receives data in this order:
        batch_idx, tuple_idx
        # batch[0][0] = first sample, the dict of data
        # batch[0][1] = first sample, the idx
        # batch[1][0] = second sample, the dict of data
        # batch[1][1] = second sample, the idx

    Returns
    -------
    sample : dictionary
        A dictionary containing tensors of the following order and shape:
            (1) input ids [batch_sz, max_seq_len]
            (2) attention masks [batch_sz, max_seq_len]
            (3) start positions [batch_sz, ]
            (4) end positions [batch_sz, ]
            (5) torch data set indices [batch_sz, ]

    '''
    input_ids = []
    attention_mask = []
    token_type_ids = []
    start = []
    end = []
    idxs = []
    for i in range(len(batch)):
        input_ids.append(batch[i][0]['input_ids'])
        attention_mask.append(batch[i][0]['attention_mask'])
        token_type_ids.append(batch[i][0]['token_type_ids'])
        start.append(batch[i][0]['start_positions'])
        end.append(batch[i][0]['end_positions'])
        idxs.append(batch[i][1])
    # package
    sample = {'input_ids': torch.as_tensor(input_ids),
              'attention_mask': torch.as_tensor(attention_mask),
              'token_type_ids': torch.as_tensor(token_type_ids),
              'start_positions': torch.as_tensor(start),
              'end_positions': torch.as_tensor(end),
              'idx': torch.as_tensor(idxs)}
    return sample

# collate fn for HuggingFace data
def collate_squad_dev(batch):
    '''
    This function collates the data emitted from the torch data set class
    and emits it as a dictionary.

    Parameters
    ----------
    batch : torch tensors
        The data loader receives data in this order:
        batch_idx, tuple_idx
        # batch[0][0] = first sample, the dict of data
        # batch[0][1] = first sample, the idx
        # batch[1][0] = second sample, the dict of data
        # batch[1][1] = second sample, the idx

    Returns
    -------
    sample : dictionary
        A dictionary containing tensors of the following order and shape:
            (1) input ids [batch_sz, max_seq_len]
            (2) attention masks [batch_sz, max_seq_len]
            (3) start positions [batch_sz, ]
            (4) end positions [batch_sz, ]
            (5) torch data set indices [batch_sz, ]

    '''
    input_ids = []
    attention_mask = []
    token_type_ids = []
    start = []
    end = []
    idxs = []
    for i in range(len(batch)):
        input_ids.append(batch[i][0]['input_ids'])
        attention_mask.append(batch[i][0]['attention_mask'])
        token_type_ids.append(batch[i][0]['token_type_ids'])
        start.append(batch[i][0]['start_positions'])
        end.append(batch[i][0]['end_positions'])
        idxs.append(batch[i][1])
    # package
    sample = {'input_ids': torch.as_tensor(input_ids),
              'attention_mask': torch.as_tensor(attention_mask),
              'token_type_ids': torch.as_tensor(token_type_ids),
              'start_positions': torch.as_tensor(start),
              'end_positions': torch.as_tensor(end),
              'idx': torch.as_tensor(idxs)}
    return sample

# collate fn for HuggingFace data
def collate_squad_score(batch):
    '''
    This function collates the data emitted from the torch data set class
    and emits it as a dictionary.

    Parameters
    ----------
    batch : torch tensors
        The data loader receives data in this order:
        batch_idx, tuple_idx
        # batch[0][0] = first sample, the dict of data
        # batch[0][1] = first sample, the idx
        # batch[1][0] = second sample, the dict of data
        # batch[1][1] = second sample, the idx

    Returns
    -------
    sample : dictionary
        A dictionary containing tensors of the following order and shape:
            (1) input ids [batch_sz, max_seq_len]
            (2) attention masks [batch_sz, max_seq_len]
            (3) start positions [batch_sz, ]
            (4) end positions [batch_sz, ]
            (5) torch data set indices [batch_sz, ]

    '''
    input_ids = []
    attention_mask = []
    token_type_ids = []
    idxs = []
    for i in range(len(batch)):
        input_ids.append(batch[i][0]['input_ids'])
        attention_mask.append(batch[i][0]['attention_mask'])
        token_type_ids.append(batch[i][0]['token_type_ids'])
        idxs.append(batch[i][1])
    # package
    sample = {'input_ids': torch.as_tensor(input_ids),
              'attention_mask': torch.as_tensor(attention_mask),
              'token_type_ids': torch.as_tensor(token_type_ids),
              'idx': torch.as_tensor(idxs)}
    return sample


# collate fn for BERT
def collate_BERT(batch):
    ''' This function packages the tokens and squeezes out the extra
    dimension.
    '''
    # turn data to tensors
    input_ids = torch.stack([torch.as_tensor(item['input_ids']) for item in batch]).squeeze(1)
    # get attn_mask
    attention_mask = torch.stack([torch.as_tensor(item['attention_mask']) for item in batch]).squeeze(1)
    # get token_type_ids
    token_type_ids = torch.stack([torch.as_tensor(item['token_type_ids']) for item in batch]).squeeze(1)
    # get labels
    labels = torch.stack([torch.as_tensor(item['labels']) for item in batch])
    # get idxs
    idxs = torch.stack([torch.as_tensor(item['idx']) for item in batch])
    # repackage
    sample = {'input_ids': input_ids,
              'attention_mask': attention_mask,
              'token_type_ids': token_type_ids,
              'labels': labels,
              'idx': idxs}
    return sample


# collate fn for H5
def collate_H5_squad(batch):
    ''' This function alters the emitted dimensions from the dataloader
    from: [batch_sz, layers, tokens, features]
    to: [layers, batch_sz, tokens, features] for the embeddings
    '''
    # turn data to tensors
    embeddings = torch.stack([torch.as_tensor(item['embeddings']) for item in batch])
    # swap to expected [layers, batch_sz, tokens, features]
    embeddings = embeddings.permute(1, 0, 2, 3)
    # get start ids
    start_ids = torch.stack([torch.as_tensor(item['start_ids']) for item in batch])
    # get end ids
    end_ids = torch.stack([torch.as_tensor(item['end_ids']) for item in batch])
    # get idxs
    idxs = torch.stack([torch.as_tensor(item['idx']) for item in batch])
    # repackage
    sample = {'embeddings': embeddings,
              'start_ids': start_ids,
              'end_ids': end_ids,
              'idx': idxs}
    return sample


# collate fn for H5
def collate_H5_GLUE(batch):
    ''' This collects the data from the H5 data set and emits them as:
     [batch_sz, layers, tokens, features]
    '''
    # turn data to tensors
    embeddings = torch.stack([torch.as_tensor(item['embeddings']) for item in batch])
    # swap to expected [layers, batch_sz, tokens, features]
    embeddings = embeddings.permute(1, 0, 2, 3)
    # get start ids
    labels = torch.stack([torch.as_tensor(item['labels']) for item in batch])
    # get idxs
    idxs = torch.stack([torch.as_tensor(item['idx']) for item in batch])
    # repackage
    sample = {'embeddings': embeddings,
              'labels': labels,
              'idx': idxs}
    return sample
