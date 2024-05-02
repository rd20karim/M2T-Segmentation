import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from datasets.vocabulary import vocabulary

class dataset_class(Dataset):
    def __init__(self, path, train=False, test=False,val=False, path_txt=None,min_freq=1,
                 filter_data=False,joint_angles=False,multiple_references=False,random_state=11,splits=True):
        super().__init__()
        self.train_ = train
        self.test_ = test
        self.val_ = val
        self.original_path = path
        self.multiple_references = multiple_references

        data = np.load(self.original_path,allow_pickle=True)
        self.splits_ids = data["splits_ids"]
        """Here with set joint angles or cartesian coordinates"""
        self.poses = data['all_motions'] if joint_angles else np.asarray(
            [xyz.reshape(-1, 21 * 3) for xyz in data['kitmld_array']], dtype=object)
        data_cls = data['old_desc']
        self.old_desc = data_cls
        assert len(data_cls) == len(self.poses)
        self.sentences =  [d for ds in data_cls for d in ds]  # flat all descriptions in order

        # read corrected descriptions
        if path_txt is not None:
            with open(path_txt, mode='r',encoding='utf-8') as ftxt:
                double_desc = ftxt.readlines()
                self.sentences = []
                for desc in double_desc:
                    ls = desc.split('\t')
                    self.sentences.append(ls[1][1:].replace("\n","")) #append the corrections and remove first space and newline symbol

        # create an index mapping
        self.indxs = list(range(len(self.poses)))
        self.map_ids_len = {i: len(ds) for i, ds in enumerate(data_cls)}
        self.map_ids_descs = {i: list(range(i, i + len(ds))) for i, ds in enumerate(data_cls)}

        correct_tokens = False  if path_txt or not filter_data and not joint_angles else True
        logging.info(f"CORRECT TOKENS {correct_tokens}")

        self.lang = vocabulary(self.sentences, correct_tokens=correct_tokens, ask_user=False)
        logging.info(f"Building vocabulary with minimum frequency : {min_freq}")
        self.lang.build_vocabulary(min_freq=min_freq)
        self.corrected_sentences = self.lang.corrected_sentences

        self.poses = self.shift_poses()
        self.poses = np.asarray([ps.reshape(-1, 21 * 3) for ps in self.poses], dtype=object)

        if filter_data and not joint_angles:
            logging.info("Normalize and save poses")
            self.poses = self.shift_poses()
            self.poses = np.asarray([ps.reshape(-1, 21 * 3) for ps in self.poses], dtype=object)

        logging.info("Convert token to numerical values")
        self.corrected_sentences_numeric = []
        for desc in self.corrected_sentences:
            self.corrected_sentences_numeric.append(
                [self.lang.token_to_idx['<sos>']] + self.lang.numericalize(desc) + [self.lang.token_to_idx['<eos>']])

        self.corrected_sentences_numeric = np.asarray(self.corrected_sentences_numeric, dtype=object)

        def dw_sample(poses, descs, s, use_multiple_samples=False):
            if use_multiple_samples:
                poses_dw = np.asarray([ps[range(k, len(ps), s)] for ps in poses for k in range(s)], dtype=object)
                sentences_dw = np.repeat(descs, s)
            else:
                start_offset = s // 2
                poses_dw = np.asarray([ps[range(start_offset, len(ps), s)] for ps in poses], dtype=object)
                sentences_dw = descs
            return poses_dw, sentences_dw

        # Motion to Language
        def sort_wlen(sequence1, sequence2):
            "Sort with respect to lengths of sequence1"
            id_lg = sorted(range(len(sequence1)), key=lambda x: len(sequence1[x]))
            sq2 = sequence2[id_lg]
            sq1 = sequence1[id_lg]
            return sq1, sq2


        assert len(self.poses) == len(self.map_ids_len)

        if not splits:
            logging.info(f"Random state {random_state} fixed to generate the same split")
            self.indxs = list(range(len(self.poses)))
            self.indx_train, self.indx_test = train_test_split(self.indxs, test_size=0.1, random_state=random_state,
                                                               shuffle=True)  # IMPORTANT SET SHUFFLE TO FALSE TO PRESERVE POSE LENGTH ORDER SAME IN DATALOADER
            self.indx_train, self.indx_val = train_test_split(self.indx_train, test_size=0.1, random_state=random_state,
                                                              shuffle=True)
        else:
            logging.info('Using official split')
            self.indx_train = np.where(np.asarray(self.splits_ids) == "train")[0]
            self.indx_val = np.where(np.asarray(self.splits_ids) == "val")[0]
            self.indx_test  = np.where(np.asarray(self.splits_ids) == "test")[0]

        #Sanity check
        len_annot = list((self.map_ids_len).values())  # take lengths of list of descriptions per motion
        cum_len = np.cumsum(len_annot)
        use_multiple_samples = False
        dw_factor = 2 # TODO !! SET THIS CORRECTLY FOR EACH DATASET
        if not multiple_references:
            # Number of annotations per motion (idmot) <=> self.map_ids_len[id_mot]
            self.X_train, self.X_val, self.X_test = [
                np.asarray([self.poses[id_mot] for id_mot in ids for _ in range(self.map_ids_len[id_mot])],
                           dtype=object)  # ids is the index of each pose
                for ids in [self.indx_train, self.indx_val, self.indx_test]]
            assert sum([self.map_ids_len[id_mot] for id_mot in self.indx_test]) == len(self.X_test)
            # MAP INDEX POSE IDS TO INDEX ANNOTATION
            # ids is the index of each pose
            self.y_train, self.y_val, self.y_test = [
                np.asarray(np.concatenate(
                    [self.corrected_sentences_numeric[0 if id_mot == 0 else cum_len[id_mot - 1]: cum_len[id_mot]]
                     for id_mot in ids]), dtype=object) for ids in [self.indx_train, self.indx_val, self.indx_test]
            ]
            assert len(self.X_train) == len(self.y_train)
            logging.info("Sort with respect to pose seq_len to reduce padding percentage --> optimize time")
            self.X_train, self.y_train = sort_wlen(self.X_train, self.y_train)
            self.X_val, self.y_val = sort_wlen(self.X_val, self.y_val)
            self.X_test, self.y_test = sort_wlen(self.X_test, self.y_test)
            logging.info("DOWN SAMPLE DATA 100Hz --> %.2f Hz" % (20 / dw_factor))
            self.X_train, self.y_train = dw_sample(self.X_train, self.y_train, s=dw_factor,
                                                   use_multiple_samples=use_multiple_samples)
            self.X_val, self.y_val = dw_sample(self.X_val, self.y_val, s=dw_factor,
                                               use_multiple_samples=use_multiple_samples)
            self.X_test, self.y_test = dw_sample(self.X_test, self.y_test, s=dw_factor,
                                                 use_multiple_samples=use_multiple_samples)
            logging.info("Samples are flattened for training (s,[r1,r2]) --> (s,r1);(s,r2)")
            logging.info("Number of flattened samples VAL: %d TRAIN: %d, TEST: %d " % (
            len(self.X_val), len(self.X_train), len(self.X_test)))

        else:
            self.X_ref_train, self.X_ref_val, self.X_ref_test = [
                np.asarray([self.poses[id_mot] for id_mot in ids], dtype=object)  # ids is the index of each pose
                for ids in [self.indx_train, self.indx_val, self.indx_test]]
            self.y_ref_train, self.y_ref_val, self.y_ref_test = [
                np.asarray([self.corrected_sentences_numeric[0 if id_mot == 0 else cum_len[id_mot - 1]: cum_len[id_mot]]
                            for id_mot in ids], dtype=object) for ids in
                [self.indx_train, self.indx_val, self.indx_test]
            ]
            self.X_ref_train, self.y_ref_train = dw_sample(self.X_ref_train, self.y_ref_train, s=dw_factor,
                                                           use_multiple_samples=use_multiple_samples)
            self.X_ref_val, self.y_ref_val = dw_sample(self.X_ref_val, self.y_ref_val, s=dw_factor,
                                                       use_multiple_samples=use_multiple_samples)
            self.X_ref_test, self.y_ref_test = dw_sample(self.X_ref_test, self.y_ref_test, s=dw_factor,
                                                         use_multiple_samples=use_multiple_samples)

            logging.info("Number  of samples VAL: %d TRAIN: %d, TEST: %d " % (
            len(self.X_ref_val), len(self.X_ref_train), len(self.X_ref_test)))

        self.Sets = [self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test] \
            if not self.multiple_references else [self.X_ref_train, self.y_ref_train, self.X_ref_val, self.y_ref_val,
                                                  self.X_ref_test, self.y_ref_test]

    def __getitem__(self, item):
        if self.train_:
            return self.Sets[0][item], self.Sets[1][item]  # (variable_sequence_length,n_joint,joint_dim)
        elif self.val_:
            return self.Sets[2][item], self.Sets[3][item]  # (variable_sequence_length,n_joint,joint_dim)
        elif self.test_:
            return self.Sets[4][item], self.Sets[5][item]  # (variable_sequence_length,n_joint,joint_dim)

    def __len__(self):
        if self.train_:
            return len(self.Sets[0])
        elif self.val_:
            return len(self.Sets[2])
        elif self.test_:
            return len(self.Sets[4])

    def shift_poses(self):
        poseswtx = self.poses
        logging.info("Std/Mean Normalization")
        temp = np.concatenate(poseswtx, axis=0).reshape(-1, 21, 3)
        x = temp[:, :, 0].flatten()
        y = temp[:, :, 1].flatten()
        z = temp[:, :, 2].flatten()

        sx, sy, sz = [np.sqrt(np.var(cord)) for cord in [x, y, z]]
        mx, my, mz = [np.mean(cord) for cord in [x, y, z]]

        normalized_poses = []
        meandata = np.expand_dims(np.array([mx, my, mz]), axis=(0, 1))
        stddata = np.expand_dims(np.array([sx, sy, sz]), axis=(0, 1))
        for k in range(len(poseswtx)):
            normalized_poses.append((poseswtx[k].reshape(-1, 21, 3) - meandata) / stddata)  #

        shift_poses = []
        for k in range(len(normalized_poses)):
            shift_poses.append(
                normalized_poses[k] - np.expand_dims(normalized_poses[k].reshape(-1, 21, 3)[0, 0, :], axis=(0, 1)))
        return np.asarray(normalized_poses, dtype=object)



if __name__ == "__main__":
    # RUN FIRST TO CREATE DATASET CORRECTION
    path = r"C:\Users\karim\PycharmProjects\HumanML3D\kit_with_splits_2023.npz"
    data = dataset_class(path, filter_data=True,min_freq=3)
