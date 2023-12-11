import numpy as np
import pandas as pd
import torchtext
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import time
from tempfile import mktemp
import os
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate import meteor_score
from nltk.corpus.reader.wordnet import  WordNetCorpusReader
def read_pred_refs(path,split=True):
    predictions = []
    references = []
    with open(path, mode="r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            predictions.append(line.split(',')[0].split(" ")) if split else predictions.append(line.split(',')[0])
            references.append([ref.split(" ") for ref in line.split(',')[1:]]) if split else references.append([ref for ref in line.split(',')[1:]])
    return predictions,references

def calculate_bleu(predictions,references,num_grams=4,single_bleu=False,smooth_method=None):
    bleu_score = torchtext.data.metrics.bleu_score(predictions, references,num_grams, weights=[1/num_grams]*num_grams)\
                                if not single_bleu else sentence_bleu(references,predictions,weights=(1/num_grams,)*num_grams,
                                                                      smoothing_function=smooth_method)
    return bleu_score

def bleu_to_df(predictions,references,smooth_method):
    scores = -np.ones((len(predictions),7),dtype=object)
    for k in range(len(predictions)):
        scores[k,0]=len(predictions[k]) # length prediction column
        scores[k,-1]=" ".join(predictions[k])
        for gr in range(1, 6):
            scores[k,gr] = calculate_bleu( predictions[k],references[k],num_grams=gr,single_bleu=True,smooth_method=smooth_method) #bleu scores columns
    df_bleu =  pd.DataFrame(scores,columns=["Length","bleu_1","bleu_2","bleu_3","bleu_4","bleu_5","prediction"])
    return df_bleu

def bleu_vs_ngram(predictions,references,n_plot=1,fig=None,color="ro--",legend_name="Joint angles",shift=0,single_figure=True):
    BLEU_scores_gram = []
    max_g = [1, 2, 3, 4, 5]
    for gr in max_g:
        bleu_score_g = calculate_bleu(predictions,references,num_grams=gr)
        BLEU_scores_gram.append(bleu_score_g)
    k = 1 if single_figure else n_plot
    fig = plt.figure(figsize=(6.4 * k, 4.8 * k)) if fig is None else fig
    ax1 = fig.axes[0] if len(fig.axes)!=0 else fig.add_subplot(int('1'+str(k)+'1'))
    ax1.set_ylim(0, 1)
    ax1.set_xticks([0,1,2,3,4,5])
    ax1.plot(max_g, BLEU_scores_gram, color,label=legend_name)
    ax1.set_ylabel("BLEU Score")
    ax1.set_xlabel("Gram number")
    #ax1.legend([legend_name])
    for x, y in zip(max_g, BLEU_scores_gram):
        ax1.text(x-0.12, y+shift, "%.3f" % y,color=color[0])
        print("\033[1;32m BLEU-",x,"--> %.2f"%(y*100))
    fig.tight_layout()
    if n_plot==1 : plt.legend();fig.savefig("BiGRU_angles_vs_cart_soft_att.png");plt.show()
    else : return  fig
    # write predictions and targets for every batch

def bleu_vs_sentence_length(predictions, references):

    logging.info("plot BLEU Score per n_gram")
    fig = bleu_vs_ngram(predictions, references,n_plot=3)

    logging.info("Plot bleu score per sentence length")
    trg = [ref[0] for ref in references] # TODO use max BLEU ON REFERENCES
    pred = predictions
    ids_sort_trg = sorted(range(len(trg)),key= lambda k: len(trg[k]))
    trg_sorted = [trg[k] for k in ids_sort_trg]
    pred_sorted = [pred[k] for k in ids_sort_trg]
    trg_lens = list(set(len(k) for k in trg_sorted))
    max_trg_len = max(trg_lens)
    pred_per_len = [[] for _ in range(max_trg_len+1)]
    trg_per_len = [[] for _ in range(max_trg_len+1)]
    for pr,tr in zip(pred_sorted,trg_sorted):
        pred_per_len[len(tr)].append(pr)
        trg_per_len[len(tr)].append([tr]) # list of reference we have one we use zero index tr =[["sentences"]]
    bleu_scores_list = []
    for k in trg_lens:
        bleu_score = calculate_bleu(pred_per_len[k],trg_per_len[k],num_grams=4)
        bleu_scores_list.append(bleu_score)
        logging.info("sentences length %d  --> BLEU Score %.3f"%(k,bleu_score))
    ax2 = fig.add_subplot(222)
    ax2.set_ylim(0,1)
    ax2.plot(trg_lens,bleu_scores_list,"go--")
    ax2.set_ylabel(f" BLEU score n_gram = {4}")
    ax2.set_xlabel(" Sentence length ")


    logging.info(" PLot Histogram ")
    ax3 = fig.add_subplot(223)
    ax3.hist([len(k) for k in trg_sorted],bins=50)
    ax3.set_title("Number of sentences")
    ax3.set_xlabel("Sentence length")
    fig.tight_layout()
    plt.show()


def semantic_bleu(predictions,references):
    return sum(meteor_score.meteor_score(references[k],predictions[k],wordnet=WordNetCorpusReader )
               for k in range(len(references)))/len(references)

if __name__=="__main__":

    path = r"C:\Users\karim\PycharmProjects\SemMotion\cartesian_mask_TRUE_TF0.7_D10_h3D_MLP.csv"
    pa, ra= read_pred_refs(path,split=True)
    bleu_score = calculate_bleu(pa,ra,num_grams=4)

    # print(path.split("\\")[-1],bleu_score)

    # # bleu_vs_sentence_length(pa,ra)
    # pa, ra= read_pred_refs("src/csv_results/BiGRU_angles_soft_att_.csv",split=True)
    # fig = bleu_vs_ngram(pa,ra,n_plot=2,shift=-0.05,single_figure=True)
    # df_bleu_angles = bleu_to_df(pa,ra,smooth_method=SmoothingFunction().method0)
    # pc,rc= read_pred_refs("src/csv_results/BiGRU_cartesian_soft_att_.csv",split=True)
    # fig = bleu_vs_ngram(pc,rc,fig=fig,legend_name="XYZ-Cartesian",color="bo--",shift=+0.05)
    # df_bleu_cartesian = bleu_to_df(pc,rc,smooth_method=SmoothingFunction().method0)
    # df_mix = pd.concat([df_bleu_angles.iloc[:,[3,-1,4]],df_bleu_cartesian.iloc[:,[4,-1,3]]],axis=1)
    #
    # meteor_angles = semantic_bleu(pa,ra)
    # meteor_cartesian = semantic_bleu(pc,rc)
