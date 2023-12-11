#from datasets.kit_txt_dataset import newkitDataset
import torch
from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import logging
import copy
#%%
#path= "/home/karim/PycharmProjects/semMotion/datasets/data_m2l_2016.npz"
#path = "/home/karim/PycharmProjects/semMotion/datasets/data_m2l_no_added_samples.npz"
#TODO remove this token from dataset http://timmcgrawtourtickets.com/

def pad_collate(batch):
    temp = list(zip(*batch))
    src = [torch.as_tensor(st) for st in temp[0]]
    target = [torch.as_tensor(st) for st in temp[1]]
    x_lens = [len(sr) for sr in src]  # get variable sequence length
    x_batch_pad = pad_sequence(src, batch_first=True, padding_value=0)  # (batch_size,TimeMax, n_joint, joint_dim)
    y_batch_pad = pad_sequence(target, batch_first=True, padding_value=0)  # (batch_size,TimeMax, n_joint, joint_dim)

    return x_batch_pad, y_batch_pad

def pad_collate_m2l(batch,return_trg_len,pad=True):
    temp = list(zip(*batch))
    poses = list(temp[0])
    targets = list(temp[1])
    x_lens = [len(pose) for pose in poses] # get variable sequence length
    # poses = torch.from_numpy(poses)
    targets_pad =  pad_sequence([torch.as_tensor(trg) for trg in targets], batch_first=True, padding_value=0) \
                    if pad else targets #(batch_size,TimeMax, n_joint, joint_dim)

    x_batch_pad =  pad_sequence([torch.as_tensor(ps) for ps in poses], batch_first=True, padding_value=0) #(batch_size,TimeMax, n_joint, joint_dim)
    if return_trg_len : return x_batch_pad, targets_pad, x_lens, [len(trg) for trg in targets]
    else : return x_batch_pad, targets_pad, x_lens


def build_data(dataset_class=None,train_batch_size=32,test_batch_size=32,min_freq=1,return_lengths=False,
               path_txt=None,return_trg_len=None,joint_angles=False,multiple_references = False,random_state=11,path=None):
    logging.info("Building dataset ... ")
    if dataset_class is None : dataset_class = newkitDataset; path_txt ="/home/karim/PycharmProjects/semMotion/datasets/sentences_autoencoder_data.csv"
    # Return trg len for jointly learning pose and text
    data_loader = dataset_class(path,path_txt=path_txt,min_freq=min_freq,joint_angles=joint_angles,
                                multiple_references = multiple_references,random_state=random_state)

    train_dataset = copy.deepcopy(data_loader)
    train_dataset.train_ = True

    test_dataset = copy.deepcopy(data_loader)
    test_dataset.test_ = True

    val_dataset = copy.deepcopy(data_loader)
    val_dataset.val_ = True

    pad_collate_func = lambda batch : pad_collate_m2l(batch,return_trg_len, pad = False if multiple_references else True) \
                                                                                    if return_lengths else pad_collate

    shuffle = False # order
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=train_batch_size,
                             batch_sampler = None,# BatchSampler(SequentialSampler(range(len(train_dataset))),batch_size = train_batch_size,drop_last=False),
                             shuffle=shuffle,collate_fn = pad_collate_func,
                             num_workers=0, pin_memory=False,drop_last=False)

    val_data_loader = DataLoader(dataset=val_dataset,batch_size=test_batch_size,
                             batch_sampler = None,# BatchSampler(SequentialSampler(range(len(test_dataset))),batch_size = test_batch_size,drop_last=False),
                             shuffle=shuffle, collate_fn = pad_collate_func ,
                             num_workers=0, pin_memory=False,drop_last=False)#, batch_size=None, sampler=None,

    test_data_loader = DataLoader(dataset=test_dataset,batch_size=test_batch_size,
                             batch_sampler = None,# BatchSampler(SequentialSampler(range(len(test_dataset))),batch_size = test_batch_size,drop_last=False),
                             shuffle=shuffle, collate_fn = pad_collate_func ,
                             num_workers=0, pin_memory=False,drop_last=False)#, batch_size=None, sampler=None,

    return train_data_loader,val_data_loader,test_data_loader
