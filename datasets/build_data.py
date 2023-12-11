import os

import numpy as np


pj = os.path.join

# TODO SELECT THE DATASET

#------------------------ FOR KIT 2022 (21 JOINTS) -----------------------------

# poses_dir = r"C:\Users\karim\PycharmProjects\HumanML3D\HumanML3D\new_joints"
# texts = r"C:\Users\karim\PycharmProjects\HumanML3D\HumanML3D\texts"


#---------------------- FOR HumanML3D (22 JOINTS) -------------------------------
abs_path = r"C:\Users\karim\PycharmProjects\HumanML3D"
poses_dir = abs_path + "new_joints/"
texts = abs_path + "texts/"

ids = []
list_desc = []
kit_poses = []
nameid = []
with open(abs_path+"train.txt") as f:
    train_ids = f.read().split("\n")
with open(abs_path+"test.txt") as f:
    test_ids = f.read().split("\n")
with open(abs_path+"val.txt") as f:
    val_ids = f.read().split("\n")

split_ids = []
for ndesc in os.listdir(texts):
    npose = ndesc.replace("txt","npy")
    try:
        kit_poses.append( np.load(pj(poses_dir,npose)))
    except FileNotFoundError: print(npose);continue
    with open(pj(texts,ndesc)) as f:
        list_desc +=[[ph.split("#")[0] for ph in f.readlines()]]
        ids += [npose.split(".")[0]]
        nameid.append(npose.split(".")[0])
        print(nameid[-1])
        if nameid[-1] in train_ids:
            split_ids.append("train")
        elif nameid[-1] in test_ids:
            split_ids.append("test")
        elif nameid[-1]  in val_ids:
            split_ids.append("val")
        else:
            print(nameid,"problem here");break

    assert npose.split(".")[0]== ndesc.split(".")[0]



#--------------- Human ML3D ------------------------

np.savez(r"C:\Users\karim\PycharmProjects\HumanML3D\all_humanML3D.npz",kitmld_array=kit_poses,old_desc=list_desc,sample_ids = nameid, splits_ids=split_ids)

# #-------------- KIT-ML -----------------------------
# np.savez(r"C:\Users\karim\PycharmProjects\HumanML3D\kit_with_splits_2023.npz",kitmld_array=kit_poses,old_desc=list_desc,sample_ids = nameid, splits_ids=split_ids)


# ids_train = np.where(np.asarray(split_ids)=="train")[0]
# ids_train_val = np.where(np.asarray(split_ids)=="val")[0]
# ids_test= np.where(np.asarray(split_ids)=="test")[0]







