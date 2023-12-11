import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from datasets.vocabulary import vocabulary


class dataset_class(Dataset):
    def __init__(self, path, train=False, test=False,val=False, path_txt=None,min_freq=1,filter_data=False,
                 joint_angles=False,multiple_references=False,random_state=11,splits=True):

        super().__init__()
        self.train_ = train
        self.test_ = test
        self.val_ = val
        self.original_path = path
        self.multiple_references = multiple_references

        data = np.load(self.original_path,allow_pickle=True)

        self.poses = data['all_motions'] if joint_angles else np.asarray([xyz.reshape(-1,22*3) for xyz in data['kitmld_array']],dtype=object)
        self.poses = self.shift_poses()
        print(len(self.poses))
        self.poses = np.asarray([ps.reshape(-1, 22 * 3) for ps in self.poses], dtype=object)
        # TODO find correspondence in os.getcwd()+"/datasets/humanML3dtext_2022.npz"
        data_cls = data["old_desc"]
        assert len(data_cls)==len(self.poses)
        self.sentences = [d for ds in data_cls for d in ds] #flat all descriptions in order

        # create an index mapping
        self.map_ids_len={i:len(ds) for i,ds in enumerate(data_cls)}
        self.map_ids_descs = {i:list(range(i,i+len(ds))) for i,ds in enumerate(data_cls)}
        if path_txt is not None:
            with open(path_txt, mode='r',encoding='utf8') as ftxt:
                double_desc = ftxt.readlines()
                self.sentences = []
                for desc in double_desc:
                    ls = desc.split('\t')
                    self.sentences.append(ls[1][1:].replace("\n","")) #append the corrections and remove first space and newline symbol

        # Cartesian coordinates
        else:
            pass
            #self.poses, self.sentences = self.read_data(path_txt=path_txt)
        if filter_data:
            logging.info("FILTER POSE WITH NO DESCRIPTION ")
            # _ , id_pose_annotated = self.filter_text(self.sentences)
            # self.poses = self.poses[id_pose_annotated]
            # self.sentences = self.sentences[id_pose_annotated]

        correct_tokens = False if path_txt or not filter_data and not joint_angles else True
        logging.info(f"CORRECT TOKENS {correct_tokens}")
        self.lang = vocabulary(self.sentences, correct_tokens=correct_tokens,ask_user=False)
        logging.info(f"Building vocabulary with minimum frequency : {min_freq}")
        self.lang.build_vocabulary(min_freq=min_freq)
        self.corrected_sentences = self.lang.corrected_sentences

        if filter_data and not joint_angles:
            logging.info("Normalize and save poses")
            #self.poses = self.shift_poses()
            self.poses = np.asarray([ps.reshape(-1,22*3) for ps in self.poses],dtype=object)

        logging.info("Convert token to numerical values")
        self.corrected_sentences_numeric = []
        for desc in self.corrected_sentences:
            self.corrected_sentences_numeric.append([self.lang.token_to_idx['<sos>']] +self.lang.numericalize(desc) + [self.lang.token_to_idx['<eos>']])

        self.corrected_sentences_numeric = np.asarray(self.corrected_sentences_numeric,dtype=object)
        def dw_sample(poses,descs,s,use_multiple_samples=False):
            if use_multiple_samples:
                poses_dw = np.asarray([ps[range(k, len(ps), s)] for ps in poses for k in range(s)],dtype=object)
                sentences_dw = np.repeat(descs, s)
            else:
                start_offset = s//2
                poses_dw = np.asarray([ps[range(start_offset, len(ps), s)] for ps in poses],dtype=object)
                sentences_dw = descs
            return  poses_dw,sentences_dw
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
            self.splits_ids=data["splits_ids"]
            logging.info('Using official split (No random state)')
            self.indx_train = np.where(np.asarray(self.splits_ids) == "train")[0]
            self.indx_val = np.where(np.asarray(self.splits_ids) == "val")[0]
            self.indx_test = np.where(np.asarray(self.splits_ids) == "test")[0]        #Sanity check

        len_annot = list((self.map_ids_len).values())  # take lengths of list of descriptions per motion
        cum_len = np.cumsum(len_annot)
        use_multiple_samples = False
       # TODO WAS CHANGED DENSE DOWN SAMPLING
        dw_factor = 2

        filter_per_src = lambda data, min_len, max_len: [j for j, y in enumerate(data) if len(y) >= min_len and len(y)<=max_len]
        filter_per_refs = lambda x, min_refs: [idx for idx, i in enumerate(x) if len(i) >= min_refs]
        filter_per_trg = lambda x,max_trg_len : [idx for idx,i in enumerate(x) if len(i)<=max_trg_len]

        max_src_len = 200 # Maximum motion duration - 10s
        min_src_len = 40 # Minimum motion duration - 2s
        min_refs = 3 # Minimum number of text descriptions for same motion

        # THIS CONSTRAINT FOUND TO BE VERY IMPORTANT TO AVOID PURE LANGUAGE MODELING
        # UNCONDITIONED ON MOTION WHICH LEDS TO OVER-FITTING

        max_trg_len = 20 # Maximum number of words from refs samples

        if not multiple_references:
            # Number of annotations per motion (idmot) <=> self.map_ids_len[id_mot]
            self.X_train, self.X_val, self.X_test  = [np.asarray([ self.poses[id_mot] for id_mot in ids for _ in range(self.map_ids_len[id_mot])],dtype=object) #ids is the index of each pose
                                                                                     for ids in [self.indx_train, self.indx_val, self.indx_test] ]
            assert sum([self.map_ids_len[id_mot] for id_mot in self.indx_test])==len(self.X_test)
            # MAP INDEX POSE IDS TO INDEX ANNOTATION
            # ids is the index of each pose
            self.y_train, self.y_val, self.y_test  = [
                                                      np.asarray(np.concatenate([self.corrected_sentences_numeric[0 if id_mot==0 else cum_len[id_mot-1] : cum_len[id_mot]]
                                                                                 for id_mot in ids]),dtype=object) for ids in [self.indx_train, self.indx_val, self.indx_test]
                                                      ]

            assert len(self.X_train)==len(self.y_train)

            # FILTER PER MOTION LENGTH
            self.filter(lambda x: filter_per_src(x,min_len=min_src_len,max_len=max_src_len),True,multiple_references)
            # FILTER PER REFS NUMBER
            self.filter(lambda x: filter_per_refs(x,min_refs=min_refs),False,multiple_references)
            # FILTER PER TRG LENGTH
            self.filter(lambda x:filter_per_trg(x,max_trg_len=max_trg_len),False,multiple_references)

            logging.info("Sort with respect to pose seq_len to reduce padding percentage --> optimize time")
            self.X_train, self.y_train = sort_wlen(self.X_train, self.y_train)
            self.X_val, self.y_val = sort_wlen(self.X_val, self.y_val)
            self.X_test, self.y_test = sort_wlen(self.X_test, self.y_test)

            logging.info("DOWN SAMPLE DATA 20Hz --> %.2f Hz" % (20 / dw_factor))
            self.X_train,self.y_train = dw_sample(self.X_train,self.y_train,s=dw_factor,use_multiple_samples=use_multiple_samples)
            self.X_val,self.y_val = dw_sample(self.X_val,self.y_val,s=dw_factor,use_multiple_samples=use_multiple_samples)
            self.X_test,self.y_test = dw_sample(self.X_test,self.y_test,s=dw_factor,use_multiple_samples=use_multiple_samples)

            logging.info("Samples are flattened for training (s,[r1,r2]) --> (s,r1);(s,r2)")
            logging.info("Number of flattened samples VAL: %d TRAIN: %d, TEST: %d " % (len(self.X_val), len(self.X_train), len(self.X_test)))

        else:
            self.X_ref_train, self.X_ref_val, self.X_ref_test = [np.asarray([ self.poses[id_mot] for id_mot in ids],dtype=object)                        #ids is the index of each pose
                                                                            for ids in [self.indx_train, self.indx_val, self.indx_test] ]
            self.y_ref_train, self.y_ref_val, self.y_ref_test  = [
                                                      np.asarray([self.corrected_sentences_numeric[0 if id_mot==0 else cum_len[id_mot-1] : cum_len[id_mot]]
                                                                                 for id_mot in ids ],dtype=object) for ids in [self.indx_train, self.indx_val, self.indx_test]
                                                      ]


            # FILTER PER MOTION LENGTH
            self.filter(lambda x: filter_per_src(x,min_len=min_src_len,max_len=max_src_len),True,multiple_references)
            # FILTER PER REFS NUMBER
            self.filter(lambda x: filter_per_refs(x,min_refs=min_refs),False,multiple_references)

            # NOT TO BE APPLIED DURING INFERENCE
            # # FILTER PER TRG LENGTH
            # self.filter(lambda x:filter_per_trg(x,max_trg_len=max_trg_len),False,multiple_references)

            logging.info("Sort with respect to pose seq_len to reduce padding percentage --> optimize time")
            self.X_ref_train, self.y_ref_train = sort_wlen(self.X_ref_train, self.y_ref_train)
            self.X_ref_val, self.y_ref_val= sort_wlen(self.X_ref_val, self.y_ref_val)
            self.X_ref_test, self.y_ref_test = sort_wlen(self.X_ref_test, self.y_ref_test)

            logging.info("DOWN SAMPLE DATA 20Hz --> %.2f Hz" % (20 / dw_factor))
            self.X_ref_train,self.y_ref_train = dw_sample(self.X_ref_train,self.y_ref_train,s=dw_factor,use_multiple_samples=use_multiple_samples)
            self.X_ref_val,self.y_ref_val = dw_sample(self.X_ref_val,self.y_ref_val,s=dw_factor,use_multiple_samples=use_multiple_samples)
            self.X_ref_test,self.y_ref_test = dw_sample(self.X_ref_test,self.y_ref_test,s=dw_factor,use_multiple_samples=use_multiple_samples)

            logging.info("Number  of samples VAL: %d TRAIN: %d, TEST: %d " % (len(self.X_ref_val), len(self.X_ref_train), len(self.X_ref_test)))
        self.Sets = [self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test] \
               if not self.multiple_references else [self.X_ref_train,self.y_ref_train,self.X_ref_val,self.y_ref_val,self.X_ref_test,self.y_ref_test]

    def filter(self,filter_function,src=False,multiple_references=False):
        if multiple_references:
            keep_ids = filter_function(self.X_ref_train if src else self.y_ref_train)
            self.y_ref_train = self.y_ref_train[keep_ids]
            self.X_ref_train = self.X_ref_train[keep_ids]

            keep_ids = filter_function(self.X_ref_test if src else self.y_ref_test)
            self.y_ref_test = self.y_ref_test[keep_ids]
            self.X_ref_test = self.X_ref_test[keep_ids]

            keep_ids = filter_function(self.X_ref_val if src else self.y_ref_val)
            self.y_ref_val = self.y_ref_val[keep_ids]
            self.X_ref_val = self.X_ref_val[keep_ids]

        else:
            keep_ids = filter_function(self.X_train if src else self.y_train)
            self.y_train = self.y_train[keep_ids]
            self.X_train = self.X_train[keep_ids]

            keep_ids = filter_function(self.X_test if src else self.y_test)
            self.y_test = self.y_test[keep_ids]
            self.X_test = self.X_test[keep_ids]

            keep_ids = filter_function(self.X_val if src else self.y_val)
            self.y_val = self.y_val[keep_ids]
            self.X_val = self.X_val[keep_ids]
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
        logging.info("Mean/Std Normalization")
        poseswtx = self.poses
        temp = np.concatenate(poseswtx,axis = 0).reshape(-1,22,3)
        x= temp[:,:,0].flatten()
        y= temp[:,:,1].flatten()
        z= temp[:,:,2].flatten()

        sx,sy,sz = [np.sqrt(np.var(cord)) for cord in [x,y,z]]
        mx,my,mz = [np.mean(cord) for cord in [x,y,z]]

        normalized_poses = []
        meandata = np.expand_dims(np.array([mx, my, mz]), axis=(0, 1))
        stddata =  np.expand_dims(np.array([sx, sy, sz]), axis=(0, 1))
        for k in range(len(poseswtx)):
            normalized_poses.append((poseswtx[k].reshape(-1,22,3) - meandata) /stddata )

        shift_poses = []
        for k in range(len(normalized_poses)):
             shift_poses.append(normalized_poses[k]-np.expand_dims(normalized_poses[k].reshape(-1,22,3)[0,0,:],axis=(0,1)))

        return  np.asarray(normalized_poses,dtype=object)

    def read_data(self,path_txt=None):
        kitmld = np.load(self.original_path, allow_pickle=True)
        poses = kitmld['kitmld_array']
        descriptions = kitmld['old_desc']
        if  path_txt:
            return poses, np.asarray(descriptions, dtype=str)
        else:
            return poses, descriptions

    def filter_text(self,descriptions):
        missed_annotation = []
        pose_annotation = []
        for idsample, descp in enumerate(descriptions):
            if len(set(descp)) <= 4 :
                missed_annotation.append(idsample)
                logging.info(f"Missed description id {idsample} : [{descp}]")
            else:
                pose_annotation.append(idsample)

        return missed_annotation, pose_annotation


if __name__ == "__main__":
    # RUN FIRST TO CREATE DATASET CORRECTION
    path = r"C:\Users\karim\PycharmProjects\HumanML3D\all_humanML3D.npz"
    data =dataset_class(path,filter_data=True,min_freq=3)
