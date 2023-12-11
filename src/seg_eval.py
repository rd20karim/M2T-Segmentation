import os
import sys
sys.path.extend([os.getcwd()])
import argparse
from matplotlib.lines import Line2D
#from matplotlib.colors import LinearSegmentedColormap as cmap
from visualizations.attention_visualization import calculate_attention_batch
from src.evaluate_m2L import  load_model_config
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
from nltk.stem import PorterStemmer
import numpy as np
import yaml
import logging
import pdb
from matplotlib.lines import Line2D


def extract_annotations(file,sep=";"):
  final_annotations ={}
  with open(file, "r") as f:
    annotations = json.load(f)
  i=0
  for annotation in annotations:
    file_id = annotation['id']
    id_annotater = annotation['annotations'][0]['completed_by']
    #print(id_annotater)
    for result in annotation['annotations'][0]['result']:
      if "meta" in result.keys():
        length = result['original_length']
        start = result['value']['start']
        end = result['value']['end']
        label = result['value']['labels'][0]
        meta = result['meta']['text'][0].split(sep)
        # print(meta)
        verb = meta[0]
        if len(meta) > 1:
          modifier = meta[1]
        else:
          modifier = ""
        final_annotations[i] = {'id': file_id, 'length': length, 'start': start, 'end': end, 'verb':verb, 'modifier': modifier, "by":id_annotater}
        i+=1
  return final_annotations

def stemming(sent):
    ps = PorterStemmer()
    ps_stem_sent = [ps.stem(words_sent) for words_sent in sent]
    return ps_stem_sent
# TODO HERE we read the file

final_annotations = extract_annotations("annotations_all.json")
df = pd.DataFrame()
df = df.from_dict(final_annotations, orient='index')


list_action_verb = set(df['verb'].to_list())

score_seg = 0
FPS = 10
words_motion_segment = []

correc = {'alk':'walk','hop':'jump','pick something up':'pick', 'waolk':'walk'}
# Ground truth segment
for dfs in df.groupby("id"):
    word_seg = {}
    # Loop over action verbs
    for i,row in enumerate(dfs[1].iterrows()):
        id_sample,length,start,end,verb,modifier,id_annoter = row[1].to_list() #row[1].to_dict()
        verb = verb+"_"+str(i)
        word_seg["id_sample"] = id_sample
        word_seg['motion_length'] = length
        word_seg[verb]=[round(start*FPS),round(end*FPS)]
    words_motion_segment.append(word_seg)

Nc =  len(words_motion_segment)
print("Number of correct annotated motions --- > ", Nc)

# Original KIT-ML

common = [1, 2, 3, 4, 6, 8, 9, 10, 12, 13, 268, 15, 16, 17, 18, 274, 20, 21, 22, 23, 281,
          27, 28, 284, 31, 32, 33, 163, 36, 37, 38, 39, 164, 168, 42, 190, 63, 191, 84, 216,
          224, 229, 105, 114]

def segmentation_score(words_motion_segment,rangeth, th=0.5,verbose=0):
    """
    :param words_motion_segment: all segment of action word
    :param method: choices {element_of, inclusion or iou}
    :param th: for inclusion or iou method
    :return:
    """
    global id_correct_samples
    id_correct_samples =[]
    scores = {"IoP":[],"IoU":[],"Element_of":[]}
    N_exp = 0
    for seg in words_motion_segment:
        id_p_1 = 0
        wrong_pred = False
        score_seg = {"IoP":0,"IoU":0,"Element_of":0}
        score_continue = {"IoP":0,"IoU":0,"Element_of":0}
        id_sample = seg["id_sample"]
        verbs = list(seg.keys())[2:]
        if id_sample in common and id_sample not in [27,114,229,268,274,284,37,216,190] :
            N_exp += 1
            index_action_words = []
            prediction = stemming(preds[id_sample - 1])
            pt = id_w2p[id_sample - 1, :len(prediction)].astype(int)
            action_words = []
            """ Compute """
            # LANGUAGE SEGMENTATION
            for idv, verb in enumerate(verbs):
                fverb = verb.split("_")[0]
                try : verb_stm = stemming([correc[fverb]])[0]
                except KeyError: verb_stm = stemming([fverb])[0]
                try:
                        # retrieving the index of verb in the prediction
                        # id_p_1 + 1 the start search index to avoid verb location redundancy
                        id_p = prediction[id_p_1 + 1:].index(verb_stm) + id_p_1 + 1
                        id_p_1 = id_p
                        index_action_words.append(id_p_1)
                        action_words.append(verb)
                except ValueError:
                    N_exp -= 1
                    wrong_pred = True

                    if verbose == 1:
                        print("id_sample-->", id_sample,"prediction -->",preds[id_sample - 1])
                        print("Wrong prediction of id sample ", "-->", id_sample, "--> verb --> ", verb_stm,
                          "--> prediction --> ", prediction)
                    #for mth in scores.keys(): scores[mth] = 0
                    break  # Stop the loop

            # MOTION PRIMITIVE SEGMENTATION
            if not wrong_pred:
                for idv in range(len(index_action_words)):
                    verb = action_words[idv]
                    D = args.D
                    id_p = index_action_words[idv]
                    ibf = index_action_words[idv+1]-1 if idv < len(index_action_words)-1 else len(prediction)-1 # index word next action

                    # start = max([0, pt[id_p] - D])
                    # end = min(pt[id_p] + D , round(seg['motion_length'] * FPS) - 1)
                    # if verbose ==1:print("[Language segment] --->"," ".join(prediction[id_p: id_p+1] ),
                    #                     f"/ [Motion segment] ---> [{start}, {end}]")

                    start = max([0, pt[id_p] - D])  #if idv!=0 else 0
                    end = min(pt[ibf] + D ,round(seg['motion_length'] * FPS) - 1)
                    if verbose ==1:print("[Language segment] --->"," ".join(prediction[id_p:ibf+1] ),f"/ [Motion segment] ---> [{start}, {end}]")
                    if start > end: print(
                        f"The start segment {start} > end segment {end} of the id_sample {id_sample} !!")
                    # P_k inferred segment of a motion primitive
                    P_k = set(range(start, end))  # [_,_[
                    # G_k the ground truth of the k_th primitive
                    G_k = set(range(seg[verb][0], seg[verb][1]))  # [_,_[
                    "----------Intersection Over Prediction--------------"
                    IoP_k = len(P_k.intersection(G_k)) / len(P_k)
                    score_seg["IoP"] = score_seg["IoP"] + 1 if IoP_k >= th else score_seg["IoP"]
                    score_continue["IoP"] += IoP_k

                    "----------Intersection Over Union--------------"
                    IoU_k = len(P_k.intersection(G_k)) / len(P_k.union(G_k))
                    score_seg["IoU"] = score_seg["IoU"] + 1 if IoU_k >= th else score_seg["IoU"]
                    score_continue["IoU"] += IoU_k

                    "----------Element of--------------"
                    score_seg["Element_of"] = score_seg["Element_of"] + 1 if pt[id_p] in G_k else score_seg["Element_of"]
                    score_continue["Element_of"]  = score_seg["Element_of"]

                id_correct_samples.append(id_sample)
                for mth in scores.keys():
                    scores[mth].append({'id_sample': id_sample - 1, 'score_seg': score_seg[mth] / len(verbs),
                                   'score_continue': score_continue[mth] / len(verbs)})
    assert len(id_correct_samples)==N_exp
    # print(f"Method {method} AVG Score SEGMENTATION th {round(th,2)}  --> ", avg_score_seg)
    # print(f"Method {method} AVG Score SEGMENTATION th {round(th,2)}  --> ", avg_score_continue)
    if verbose==1: print("Number of correct annotated motions and predictions ---> ", N_exp)
    return scores




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,help="Path of model weights not used if the config file is passed")
    parser.add_argument("--dataset_name",type=str,default="kit2016",choices=["h3D","kit","kit2016"])
    parser.add_argument("--config",type=str,default="./configs/local_rec/MLP.yaml")
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--input_type",type=str,default="cartesian",choices=["cartesian","angles"])
    parser.add_argument("--multiple_references",type=bool,default=True,help="Specify evaluation mode use flattened references or all at one")
    parser.add_argument("--encoder_type",type=str,default="MLP",choices=["GRU","BiGRU","MLP","deep-MLP"])
    parser.add_argument("--attention_type",type=str,default="local_recurrent",choices=["local_recurrent","local","soft"])
    parser.add_argument("--mask",type=bool,default=True,choices=[True,False])
    parser.add_argument("--name_file",type=str,default="sent_pred",help="File name of predicted sentences")
    parser.add_argument("--subset",type=str,default="test",help="Subset on which evaluating the data",choices=["test","val"])
    parser.add_argument("--hidden_dim", type=int, default=128, help='hidden dimension of the encoder layers for the MLP')
    parser.add_argument("--hidden_size", type=int, default=64, help='hidden size of the decoder and encoder output dimension')
    parser.add_argument("--embedding_dim", type=int, default=64, help='embedding dimension of words')
    parser.add_argument("--min_freq", type=int, default=3, help='minimum word frequency to keep in the dataset')
    parser.add_argument("--batch_size", type=int, default=512, help='Batch size should be >= to length of data for Corpus BLEU score')
    parser.add_argument("--beam_size", type=int, default=1, help='beam search width')
    parser.add_argument("--D",type=int,default=5,help="Half window length")
    parser.add_argument("--random_state",type=int,default=11,help="random_state")
    parser.add_argument("--scale",default=1.,help="specify a float value or set True for automatic selection")

    args = parser.parse_args()

    list_encoders = [args.encoder_type] # ["BiGRU",'Deep-MLP', 'MLP', 'GRU']
    args.config = "../configs/local_rec/MLP.yaml"  # "./configs/local_rec/BiGRU.yaml","./configs/local_rec/deep-MLP.yaml","./configs/local_rec/MLP.yaml","./configs/local_rec/GRU.yaml"]

    seg_df_list = []
    device = torch.device(args.device)
    logging.info(f" INPUT {args.input_type} Encoder {args.encoder_type}")


    with open(args.config, 'r') as f:
        choices = yaml.load(f, Loader=yaml.Loader)

    default_arg = vars(args)
    parser.set_defaults(**choices)
    args = parser.parse_args()

    loaded_model, train_data_loader, test_data_loader,val_data_loader = load_model_config(device=device,args=args)
    data_loader = test_data_loader
    att_batch, trgs, preds, lens, src_poses, id_w2p = calculate_attention_batch(data_loader=data_loader,
                                                                                loaded_model=loaded_model,
                                                                                vocab_obj=data_loader.dataset.lang,
                                                                                indx_batch=0, word=None,
                                                                                multiple_references=args.multiple_references,device=device)
    segmentation_results = {}
    segmentation_results['Threshold'] = []
    ### INITIALIZATION
    for method in ["IoP", "IoU"]:
        for type in ["_bin","_cont"]:
            segmentation_results[method+type] = []
    #### ----------------------------------------- ####
    list_th = list(np.arange(0,1.05,.05))

    for th in list_th:
        segmentation_results['Threshold'] += [th]
        scores = segmentation_score(words_motion_segment,None,th=th,verbose=1 if th==list_th[-1] else 0)
        for method in ["IoP", "IoU"]:
            score_bin = np.mean([sc['score_seg'] for sc in scores[method]])
            score_continue = np.mean([sc['score_continue'] for sc in scores[method]])
            segmentation_results[method+"_bin"] +=\
                [score_bin]
            segmentation_results[method+"_cont"] +=\
                [score_continue]

    seg_el = np.mean([sc['score_continue'] for sc in scores["Element_of"]])
    segmentation_results["Element_of"] = [seg_el]*len(list_th)

    seg_df = pd.DataFrame()
    seg_df = seg_df.from_dict(segmentation_results, orient='index')
    seg_df_list.append(seg_df)

custom_lines = [Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="Blue", lw=4),
                Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="red", lw=4)]
def plot(method,start=1):
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xticks(np.arange(0, 1.1, .1))
    ax.set_yticks(np.arange(0, 1.1, .1))

    for j, dr in enumerate(seg_df_list[start:]):
        dr = dr.T
        dr[["Threshold", f"{method}_cont", f"{method}_bin"]].plot(x='Threshold', ax=ax, style=
        [["c--", "c--"], ["b--", "b"], ["g--", "g"], ["r--", "r"]][start:][j],lw=[3,1,1,1][j],alpha=[0.8,1,1,1][j])
    ax.legend(custom_lines[start:], ["BiGRU", 'Deep-MLP', 'MLP', 'GRU'][start:])
    plt.ylabel("Segmentation score",fontsize=13)
    plt.xlabel("Threshold",fontsize=13)

    plt.title(f"Score evolution for {method} method",fontsize=13)
    plt.savefig(f"segScore_vs_threshold_{method}.png")

    plt.show()

value = 0
plot(method = "IoP",start=value)
plot(method = "IoU",start=value)
# plot(method = "Element_of",start=value)

