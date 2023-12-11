import sys,os
sys.path.extend([os.getcwd()])
import torch
from src.evaluate_m2L import  load_model_config
import matplotlib.pyplot as plt
from visualizations.attention_visualization import calculate_attention_batch
from visualizations.subplot_3d_with_txt import SubplotAnimation
#from visualizations.frozen_motion import SubplotAnimation
from matplotlib import rc
import seaborn as sns
from matplotlib import patches
import os
import argparse
import numpy as np
import pandas as pd
from torchtext.data.metrics import bleu_score
import yaml
def softmax(x,axis=0):
    return(np.exp(x))/np.exp(x).sum(axis=axis)
def froze_gif(att_w,src_poses,start_pad,pred,sample_id,name_directory=None,idxs=None,ref=None,pts=None):
    W = att_w[:start_pad+1,:len(pred)]
    print(W.shape)
    idxs = np.argmax(W,axis=0) if idxs is None else idxs
    Wif = W.copy() ; Wif[Wif==0] = -np.inf
    att_sample = softmax(100*Wif,axis=0)
    max_per_word = np.max(att_sample,axis=0)
    predictions = preds[id_sample]
    aw= len(predictions) if len(predictions) <= 5 else len(predictions) // 2 + [0, 1][len(predictions) % 2]
    h = (2 if len(predictions)>5 else 1)
    print(aw,h)
    fig = plt.figure(figsize=(4.8 * aw, 4.8 * h))

    axes = [fig.add_subplot(h,aw, a + 1,projection='3d') for a in range((len(predictions)))]
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.subplots_adjust(wspace=0)
    for kw,pt in enumerate(pts[sample_id, :len(predictions)].astype(np.int)):
        #ax = fig.add_subplot(111, projection="3d")
        fzmotion = SubplotAnimation(src_poses, frames=range(src_poses.shape[0]),
                                    use_kps_3d=range(21),down_sample_factor=1,idxs=idxs,pred=pred,ref=ref,ax=axes[kw])
        for i in range(max(pt - 5, 0), min(pt + 5,W.shape[0]-1)):
            print(i,"-->", att_sample[i,kw]/max_per_word[kw])
            fzmotion.draw_frame(i,kw,alpha=att_sample[i,kw]/max_per_word[kw])
    name_file = name_directory+f"/fz_sample_{sample_id}_.png"
    #fig.tight_layout()
    fig.savefig(name_file,bbox_inches="tight")
    print("Animation saved in ", os.getcwd() + "/" + name_file)

def froze_segment_gif(att_w,src_poses,start_pad,pred,sample_id,name_directory=None,idxs=None,ref=None,pts=None,kms=None):
    global  att_sample
    W = att_w[:start_pad+1,:len(pred)]
    print(W.shape)
    idxs = np.argmax(W,axis=0) if idxs is None else idxs
    Wif = W.copy() ; Wif[Wif==0] = -np.inf
    att_sample = Wif
    max_per_word = np.max(att_sample,axis=0)
    predictions = preds[id_sample]
    N_segment = len(kms)
    aw = N_segment
    h = 1
    print(aw,h)
    fig = plt.figure(figsize=(4.8 * aw, 4.8 * h))

    axes = [fig.add_subplot(h,aw, a + 1,projection='3d') for a in range(N_segment)]
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.subplots_adjust(wspace=0)
    for kw in range(N_segment):
        end_index = kms[kw+1] if kw < len(kms)-1 else len(predictions)
        end_frame = pts[sample_id, :len(predictions)].astype(np.int)[end_index-1]
        pt = pts[sample_id, :len(predictions)].astype(np.int)[kms[kw]]
        # ax = fig.add_subplot(111, projection="3d")
        lang_seg = pred[:]
        lang_seg[kw] = " ".join(pred[kms[kw]: end_index])
        fzmotion = SubplotAnimation(src_poses, frames=range(src_poses.shape[0]),
                                    use_kps_3d=range(21), down_sample_factor=1, idxs=idxs, pred=lang_seg, ref=ref,
                                    ax=axes[kw])
        new_att = np.mean(W[:, kms[kw]: end_index],axis=-1)
        new_att = softmax(100 * new_att, axis=0)
        max_ = np.max(new_att, axis=-1)

        for i in range(max(pt - 5, 0), min(end_frame + 5, W.shape[0])):
            print(i, "-->", new_att[i] / max_)
            fzmotion.draw_frame(i, kw, alpha=new_att[i]/max_)
    name_file = name_directory+f"/fz_sample_{sample_id}_.png"
    #fig.tight_layout()
    fig.savefig(name_file,bbox_inches="tight")
    print("Animation saved in ", os.getcwd() + "/" + name_file)


def map_pose2concept(att_w,src_poses,start_pad,pred,sample_id,name_directory=None,
                     idxs=None,ref=None,_format='.mp4',save=False,dataset_name='kit2016'):
    W = att_w[:start_pad+1,:len(pred)]
    idxs = np.argmax(W,axis=0) if idxs is None else idxs
    #print(idxs)
    Frames_indxs = np.arange(start_pad)
    #idxs = np.round(Frames_indxs@W)
    n_joint = 21 if "kit" in dataset_name else 22
    ani = SubplotAnimation(src_poses, frames=Frames_indxs,use_kps_3d=range(n_joint),sample=sample_id,
                           down_sample_factor=1,idxs=idxs ,pred=pred,ref=ref,dataset_name=dataset_name)
    if save:
        if not  os.path.isdir(name_directory) : os.makedirs(name_directory)
        name_file = name_directory+f"/attention_sample_{sample_id}"+_format
        ani.save(name_file)
        print("Animation saved in ", os.getcwd() + "/" + name_file)
    plt.rc('animation', html='jshtml')
    return ani

def save_attention_figs(limit):
    global df
    for id_sample in range(len(lens[:limit])):
        att_w = att_batch[:,id_sample,:]
        prediction = preds[id_sample]
        start_pad = lens[id_sample]
        id_best_ref = np.argmax([bleu_score([prediction],[[ref]]) for ref in trgs[id_sample]])
        trg = trgs[id_sample][id_best_ref]
        df = pd.DataFrame(data=att_w[:start_pad,:len(prediction) ],columns=prediction)
        fig,ax = plt.subplots(figsize=(len(df)//3,len(prediction)//2))
        fsz= 17
        min_att = df[df>.005].idxmin(axis=0).values
        max_att = df.idxmax(axis=0).values
        ax = sns.heatmap((df*100).transpose(),annot=False,fmt=".0f",cmap="viridis",linewidths=0.009,ax=ax)
        iw = 0
        for indx_min,indx_max in zip(min_att,max_att):
            ax.add_patch(patches.Rectangle((indx_min,iw), 1, 1, fill=False, edgecolor='orange', lw=2))
            ax.add_patch(patches.Rectangle((indx_max,iw), 1, 1, fill=False, edgecolor='red', lw=3))
            iw += 1
        ax.set_xlabel(f" Frame index ", fontsize=fsz)
        ax.set_ylabel(" Predicted words ", fontsize=fsz, color='green')
        twin_ax = ax.twinx()
        twin_ax.yaxis.tick_right()
        twin_ax.set_yticks([k+0.5 for k in range(len(prediction))])
        twin_ax.set_yticklabels(id_w2p[id_sample,:len(prediction)])
        twin_ax.set_ylim(ax.get_ylim())
        # Maximum attention positions
        twin_ax.set_ylabel("Alignment position p_t", fontsize=fsz, color='green')
        ax.tick_params(labelbottom=True, labeltop=True,bottom=True, top=True)
        ax.set_title(f" {' '.join(trg) } ", fontsize=fsz, color='red')
        fig.tight_layout()
        os.makedirs(save_dir,exist_ok=True)
        fig.savefig(save_dir+f"/attention_sample_{id_sample}")

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
    parser.add_argument("--n_map",type=int,required=True,help="Number of attention map to generate")
    parser.add_argument("--n_gifs",type=int,required=True,help="Number of animation to generate")
    parser.add_argument("--save_results",type=str,required=True,help="Directory where to save generated plots")
    parser.add_argument("--D",type=int,default=5,help="Half window length")
    parser.add_argument("--random_state",type=int,default=11,help="random_state")
    parser.add_argument("--scale",default=1.,help="specify a float value or set True for automatic selection")

    args = parser.parse_args()
    # Update args parser
    with open(args.config,'r') as f:
        choices = yaml.load(f,Loader=yaml.Loader)
    default_arg = vars(args)
    parser.set_defaults(**choices)
    args = parser.parse_args()

    device = torch.device(args.device)
    input_type = args.input_type

    loaded_model, train_data_loader, test_data_loader,val_data_loader = load_model_config(device=device,args=args)

    data_loader = locals()[args.subset+"_data_loader"]

    indx_batch = -1
    att_batch, trgs, preds, lens, src_poses,id_w2p = calculate_attention_batch(data_loader=data_loader, loaded_model=loaded_model,
                                                                                vocab_obj=data_loader.dataset.lang,indx_batch=indx_batch,
                                                                                word=None,multiple_references=args.multiple_references,
                                                                                device=device)

    save_dir = args.save_results
    if args.n_map:
        save_attention_figs(limit=args.n_map)
    for ll, id_sample in enumerate(range(99,args.n_gifs)):
      pred = preds[id_sample]
      id_best_ref = np.argmax([bleu_score([pred], [[ref]]) for ref in trgs[id_sample]])
      trg = trgs[id_sample][id_best_ref]
      trg = ' '.join(trg)
      start_pad = lens[id_sample]
      # Shifting [ idxs = id_w2p[id_sample,:len(pred)]-D ]
      map_pose2concept(att_batch[:,id_sample,:],np.asarray(src_poses[:start_pad,:,:].cpu()),start_pad,pred,
                       sample_id = id_sample,name_directory=args.save_results,ref=trg,
                       idxs=(id_w2p[id_sample,:len(pred)]).astype(int),dataset_name=args.dataset_name,save=True,_format=".gif")
      plt.close()