import os
from visualizations.attention_visualization import calculate_attention_batch
from src.evaluate_m2L import  load_model_config
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
from nltk.stem import PorterStemmer
import numpy as np

import argparse,yaml

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


# # GET ID SAMPLES
# indx_test = data_loader.dataset.indx_test
# # ID OF LABELED DATA
# id_labeled = df['id'].to_list()
# # MAPPING TO IDS IN DATASET
# ids_in_dataset = [indx_test.index(id_l) for id_l in id_labeled]

def stemming(sent):
    ps = PorterStemmer()
    ps_stem_sent = [ps.stem(words_sent) for words_sent in sent]
    return ps_stem_sent

# TODO HERE we read the file
final_annotations = extract_annotations(r"C:\Users\karim\PycharmProjects\SemMotion\src\annotations_all.json")
df = pd.DataFrame()
df = df.from_dict(final_annotations, orient='index')

list_action_verb = set(df['verb'].to_list())

score_seg = 0
FPS = 10
words_motion_segment = []
# df = pd.concat([df_rd,df_ju,df_and])

correc = {'alk':'walk','hop':'jump','pick something up':'pick','waolk':'walk'}
# Ground truth segment
for dfs in df.groupby("id"):
    word_seg = {}
    # Loop over action verbs
    for row in dfs[1].iterrows():
        id_sample,length,start,end,verb,modifier,id_annoter = row[1].to_list() #row[1].to_dict()
        word_seg["id_sample"] = id_sample
        word_seg['motion_length'] = length
        word_seg[verb]=[round(start*FPS),round(end*FPS)]
    words_motion_segment.append(word_seg)

Nc =  len(words_motion_segment)
print("Number of correct annotated motions --- > ", Nc)


def segmentation_score(words_motion_segment,method="IoP", th=0.5):
    """
    :param words_motion_segment: all segment of action word
    :param method: choices {element_of, inclusion or iou}
    :param th: for inclusion or iou method
    :return:
    """
    global id_correct_samples
    id_correct_samples =[]
    scores = []
    N_exp =0
    for seg in words_motion_segment:
        N_exp += 1
        id_p_1 = 0
        wrong_pred = False
        score_seg = 0
        id_sample = seg["id_sample"]
        verbs = list(seg.keys())[2:]
        if id_sample in common and id_sample not in [27,114,229,268,274,284] :
            for idv, verb in enumerate(verbs):
                try : verb_stm = stemming([correc[verb]])[0]
                except KeyError: verb_stm = stemming([verb])[0]
                score_continue = 0
                prediction = stemming(preds[id_sample - 1])
                pt = id_w2p[id_sample - 1, :len(prediction)].astype(int)
                try:
                    # retrieving the index of verb in the prediction
                    # id_p_1 + 1 the start search index to avoid verb location redundancy
                    id_p = prediction[id_p_1 + 1:].index(verb_stm) + id_p_1 + 1
                    D = 5 # for the MLP-GRU D=5
                    start = max([0, pt[id_p] - D])
                    end = min(pt[id_p] + D if idv != len(verbs) - 1 else pt[-1] + D,round(seg['motion_length']*FPS))
                    # print(verb)
                    # print(start, end)
                    if start > end: print(f"The start segment {start} > end segment {end} of the id_sample {id_sample} !!")
                    # P_k inferred segment of a motion primitive
                    P_k = set(range(start, end)) # [_,_[
                    # G_k the ground truth of the k_th primitive
                    G_k = set(range(seg[verb][0], seg[verb][1])) # [_,_[
                    id_p_1 = id_p

                    if method=="IoP":
                        IoP_k = len(P_k.intersection(G_k)) / len(P_k)
                        score_seg = score_seg + 1 if IoP_k >= th else score_seg
                        score_continue += IoP_k
                    elif method=="IoU":
                        IoU_k = len(P_k.intersection(G_k)) / len(P_k.union(G_k))
                        score_seg = score_seg + 1 if IoU_k >= th else score_seg
                        score_continue += IoU_k
                    elif method=="Element_of":
                        score_seg = score_seg + 1 if pt[id_p] in G_k else score_seg
                    # TODO Second method
                except ValueError:
                    N_exp -= 1
                    wrong_pred = True
                    #print("id_sample-->", id_sample,"prediction -->",preds[id_sample - 1])
                    print("Wrong prediction of id sample ", "-->", id_sample, "--> verb --> ", verb_stm,"--> prediction --> ",prediction)
                    score_seg = 0
                    break  # Stop the loop

            if wrong_pred:
                continue # Move to the next sample
            else : id_correct_samples.append(id_sample)

            scores.append({'id_sample': id_sample - 1, 'score_seg': score_seg / len(verbs),
                           'score_continue':score_continue / len(verbs)})
            # TODO LANGUAGE SEGMENTATION
            # TODO MOTION PRIMITIVE SEGMENTATION
        else:
            N_exp -= 1
            continue

    assert len(id_correct_samples)==N_exp
    avg_score_seg = np.mean([sc['score_seg'] for sc in scores])
    avg_score_continue = np.mean([sc['score_continue'] for sc in scores])
    #print(f"Method {method} AVG Score SEGMENTATION th {round(th,2)}  --> ", avg_score_seg)
    #print(f"Method {method} AVG Score SEGMENTATION th {round(th,2)}  --> ", avg_score_continue)
    #print("Number of correct annotated motions and predictions ---> ", N_exp)
    return avg_score_seg,avg_score_continue,scores



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default="./models/local_rec/MLP_cartesian_loc_rec_D=5_mask=True.zip",help="Path of model weights not used if the config file is passed")
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

    common = [1, 2, 3, 4, 6, 8, 9, 10, 12, 13, 268, 15, 16, 17, 18, 274, 20, 21, 22, 23, 281,
              27, 28, 284, 31, 32, 33, 163, 36, 37, 38, 39, 164, 168, 42, 190, 63, 191, 84, 216,
              224, 229, 105, 114]

    paths = ["./models/local_rec/BiGRU_loc_rec.zip",
            "./models/local_rec/deep_MLP_cartesian_loc_rec_D=5_mask=True.zip",
            "./models/local_rec/MLP_cartesian_loc_rec_D=5_mask=True.zip",
            "./models/local_rec/GRU_loc_rec.zip"]

    # Build data
    _, train_data_loader, test_data_loader, val_data_loader = load_model_config(args, args.device)

    seg_df_list = []
    for k,path in enumerate(paths):

        multiple_references = True
        args.encoder_type = ["BiGRU","deep-MLP","MLP","GRU"][k]
        args.config = "./configs/local_rec/"+args.encoder_type+".yaml"

        # Update args parser
        with open(args.config, 'r') as f:
            choices = yaml.load(f, Loader=yaml.Loader)
        default_arg = vars(args)
        parser.set_defaults(**choices)
        args = parser.parse_args()

        loaded_model, _,_,_ = load_model_config(args,args.device,load_data=False,input_dim=train_data_loader.dataset.lang.vocab_size_unk)

        data_loader = test_data_loader
        att_batch, trgs, preds, lens, src_poses, id_w2p = calculate_attention_batch(data_loader=data_loader,
                                                                                    loaded_model=loaded_model,
                                                                                    vocab_obj=data_loader.dataset.lang,
                                                                                    indx_batch=0, word=None,
                                                                                    multiple_references=multiple_references,
                                                                                    device=args.device)

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
            for method in ["IoP", "IoU"]:
                score_bin, score_continue, scores = segmentation_score(words_motion_segment,method,th=th)
                segmentation_results[method+"_bin"] +=\
                    [score_bin]
                segmentation_results[method+"_cont"] +=\
                    [score_continue]

        seg_el,_,scores_el = segmentation_score(words_motion_segment,"Element_of",th=0)
        segmentation_results["Element_of"] = [seg_el]*len(list_th)

        seg_df = pd.DataFrame()
        seg_df = seg_df.from_dict(segmentation_results, orient='index')
        seg_df_list.append(seg_df)
    from matplotlib.lines import Line2D
    #from matplotlib.colors import LinearSegmentedColormap as cmap
    custom_lines = [Line2D([0], [0], color="yellow", lw=8,linestyle="-"),
                    Line2D([0], [0], color="Blue", lw=4),
                    Line2D([0], [0], color="green", lw=4),
                    Line2D([0], [0], color="red", lw=4)]



    method = "IoP"
    ax = plt.gca()
    ax.set_xticks(np.arange(0,1.1,.1))
    ax.set_yticks(np.arange(0,1.1,.1))

    for j,dr in enumerate(seg_df_list[1:]):
        dr = dr.T
        dr[["Threshold",f"{method}_cont",f"{method}_bin"]].plot(x='Threshold',ax=ax,style=[["y--","y"],["b--","b"],["g--","g"],["r--","r"]][1:][j])
    ax.legend(custom_lines[:], ["BiGRU",'Deep-MLP', 'MLP', 'GRU'][:])

    plt.ylabel("Segmentation score")
    plt.title(f"Score evolution for {method} method")
    plt.savefig(f"segScore_vs_threshold_{method}.png")

    plt.show()

    method = "Element_of"
    ax = plt.gca()
    ax.set_xticks(np.arange(0,1.1,.1))
    ax.set_yticks(np.arange(0,1.1,.1))

    for j,dr in enumerate(seg_df_list[1:]):
        dr = dr.T
        dr[["Threshold",f"{method}"]].plot(x='Threshold',ax=ax,style=[["y--","y"],["b--","b"],["g--","g"],["r--","r"]][1:][j])
    ax.legend(custom_lines[:], ["BiGRU",'Deep-MLP', 'MLP', 'GRU'][:])

    plt.ylabel("Segmentation score")
    plt.title(f"Score evolution for {method} method")
    plt.savefig(f"segScore_vs_threshold_{method}.png")

    plt.show()
