import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from datasets.vocabulary_2016 import vocabulary

classes_names = ["walk", "push_recovery", "run", "kick", "throw", "bow", "squat", "punch", "stomp", "jump",
                 "golf", "tennis", "wave", "play-guitar", "play-violin", "turn", "bend", "pick",
                 "danc"]  # danc to retrieve dancing one push-recovery is not counted

incorrect_desc = "Tim McGraw is a famous country singer, so don't miss the possibility to visit <a href=http://timmcgrawtourtickets.com/>Tim McGraw tour bus</a>"


class dataset_class(Dataset):
    def __init__(self, path, train=False, test=False, val=False, path_txt=None, min_freq=1, filter_data=False,
                 joint_angles=False, multiple_references=False, random_state=11,max_trg_len=1000):

        super().__init__()
        self.train_ = train
        self.test_ = test
        self.val_ = val
        self.original_path = path
        self.multiple_references = multiple_references
        data = np.load(self.original_path,allow_pickle=True)

        # Filter data first to get the txt file of descriptions
        assert path_txt or filter_data
        # Read corrected descriptions and their motions
        if path_txt is not None:
            with open(path_txt, mode='r') as ftxt:
                double_desc = ftxt.readlines()
                self.sentences = []
                for desc in double_desc:
                    ls = desc.split('\t')
                    self.sentences.append(
                        ls[1][1:].replace("\n", ""))  # append the corrections and remove first space and newline symbol
                # filter_data = True
            self.poses = data['all_motions'] if joint_angles else np.asarray(
                [xyz.reshape(-1, 21 * 3) for xyz in data['normalized_poses']], dtype=object)
            id_st = [ds[0] for ds in data['descriptions']].index(incorrect_desc)
            #data_cls = data['descriptions']
            data_cls = np.delete(data['descriptions'],id_st)  # remove all reference (there are one reference) of this movement
            self.poses = np.delete(self.poses, id_st)
            assert len(data_cls) == len(self.poses)
        else:
            self.poses, data_cls = self.read_data(path_txt=path_txt)
            self.old_desc = data['descriptions']

        if filter_data:
            logging.info("FILTER POSE WITH NO DESCRIPTION ")
            _, id_pose_annotated = self.filter_text(data_cls)
            #id_pose_annotated = range(len(data_cls))
            self.poses = self.poses[id_pose_annotated]
            refs = data_cls[id_pose_annotated]
            # flat all descriptions in order
            self.sentences = [d for ds in refs for d in ds]

        # create an index mapping
        self.indxs = list(range(len(self.poses)))
        self.map_ids_len = {i: len(ds) for i, ds in enumerate(data_cls)}
        self.map_ids_descs = {i: list(range(i, i + len(ds))) for i, ds in enumerate(data_cls)}

        correct_tokens = False if path_txt or not filter_data and not joint_angles else True
        logging.info(f"CORRECT TOKENS {correct_tokens}")
        self.lang = vocabulary(self.sentences, correct_tokens=correct_tokens, ask_user=False)
        logging.info(f"Building vocabulary with minimum frequency : {min_freq}")
        self.lang.build_vocabulary(min_freq=min_freq)
        self.corrected_sentences = self.lang.corrected_sentences

        if filter_data and not joint_angles:
            logging.info("Normalize and save poses")
            self.poses = self.shift_poses(save=True)
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

        logging.info(f"Random state {random_state} fixed to generate the same split")

        assert len(self.poses) == len(self.map_ids_len)
        self.indx_train, self.indx_test = train_test_split(self.indxs, test_size=0.1, random_state=random_state,
                                                           shuffle=True)  # IMPORTANT SET SHUFFLE TO FALSE TO PRESERVE POSE LENGTH ORDER SAME IN DATALOADER
        self.indx_train, self.indx_val = train_test_split(self.indx_train, test_size=0.1, random_state=random_state,
                                                          shuffle=True)

        # Sanity check
        len_annot = list((self.map_ids_len).values())  # take lengths of list of descriptions per motion
        cum_len = np.cumsum(len_annot)
        use_multiple_samples = False
        dw_factor = 10
        if not multiple_references:
            # Number of annotations per motion (idmot) <=> self.map_ids_len[id_mot]
            logging.info("Samples are flattened for training (s,[r1,r2]) --> (s,r1);(s,r2)")
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
            logging.info("DOWN SAMPLE DATA 100Hz --> %.2f Hz" % (100 / dw_factor))
            self.X_train, self.y_train = dw_sample(self.X_train, self.y_train, s=dw_factor,
                                                   use_multiple_samples=use_multiple_samples)
            self.X_val, self.y_val = dw_sample(self.X_val, self.y_val, s=dw_factor,
                                               use_multiple_samples=use_multiple_samples)
            self.X_test, self.y_test = dw_sample(self.X_test, self.y_test, s=dw_factor,
                                                 use_multiple_samples=use_multiple_samples)

            #ignored_samples = [262, 449, 391, 92, 10, 235, 314, 164, 295, 465]
            # # Filter per trg length and noisy sample
            # keep_ids = [j for j,y in enumerate(self.y_val) if len(y)<=max_trg_len ] #and j not in ignored_samples
            # self.y_val = self.y_val[keep_ids]
            # self.X_val = self.X_val[keep_ids]
            #
            # #Filter per trg length
            # keep_ids = [j for j,y in enumerate(self.y_train) if len(y)<=max_trg_len ]
            # self.y_train = self.y_train[keep_ids]
            # self.X_train = self.X_train[keep_ids]

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

    def shift_poses(self, save=False):
        logging.info("Normalizing and saving data ... to ./data_m2l_2016.npz")
        poseswtx = self.poses
        if not save:
            m, n, u, v, z, p = (3794.838833287814, 2220.888381693713, 4303.78878057613,
                                -3165.565960636252, -331.428512044666, -3983.072138900624)
        # Find scale Over all kitmld dataset
        else:
            temp = np.concatenate(poseswtx, axis=0).reshape(-1, 21, 3)
            x = temp[:, :, 0].flatten()
            y = temp[:, :, 1].flatten()
            z = temp[:, :, 2].flatten()
            m, n, u, v, z, p = max(x), max(y), max(z), min(x), min(y), min(z)

        mindata = np.expand_dims(np.array([v, z, p]), axis=(0, 1))
        maxdata = np.expand_dims(np.array([m, n, u]), axis=(0, 1))
        normalized_poses = []
        for k in range(len(poseswtx)):
            normalized_poses.append(2 * (poseswtx[k].reshape(-1, 21, 3) - mindata) / (maxdata - mindata) - 1)  #
        # check normalization process
        # shift all motion to zero center will be the start of the root joint of each motion
        shift_poses = []
        for k in range(len(normalized_poses)):
            shift_poses.append(
                normalized_poses[k] - np.expand_dims(normalized_poses[k].reshape(-1, 21, 3)[0, 0, :], axis=(0, 1)))
        if save:
            np.savez("./data_m2l_2016.npz", normalized_poses=np.asarray(normalized_poses, dtype=object), descriptions=self.old_desc)
        return np.asarray(normalized_poses, dtype=object)

    def read_data(self, path_txt=None):
        kitmld = np.load(self.original_path, allow_pickle=True)
        poses = kitmld['kitmld_array']
        descriptions = kitmld['descriptions']
        if path_txt:
            return poses, np.asarray(descriptions, dtype=str)
        else:
            return poses, descriptions

    def filter_text(self, descriptions):
        missed_annotation = []
        pose_annotation = []
        for idsample, descp in enumerate(descriptions):
            if len(descp) <= 0:
                logging.info(f"Missed description id {idsample} : [{descp}]")
                missed_annotation.append(idsample)
            elif self.poses[idsample].shape[0] / 100 >= 30:
                    logging.info(f"Pose longer than 30s")
                    missed_annotation.append(idsample)
            else:
                pose_annotation.append(idsample)

        return missed_annotation, pose_annotation


if __name__ == "__main__":
    # RUN FIRST TO CREATE DATASET CORRECTION
    path = ".\kitmld_anglejoint_2016_30s_final_cartesian.npz"  # "/home/karim/PycharmProjects/animationplot/data/kit_wtx_normalized_poses.npz"
    data = kitm2l(path, filter_data=True)
