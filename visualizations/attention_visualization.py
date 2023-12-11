import torch
import torchtext
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib_inline
from torch.nn.utils.rnn import pad_sequence
from datasets.visualization import decode_predictions_and_compute_bleu_score
def visualize_attention(input_sentence, output_words, attentions ,fig ,i ,fsz=20):
    # Set up figure with colorbar
    # fig = plt.figure()
    fig.set_facecolor("white")
    ax = fig.add_subplot(i)
    cax = ax.matshow(attentions  )  # , cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    print("input " ,len(input_sentence.split(' ')))
    print("output" ,len(output_words.split(' ')))
    print("attention shape" ,attentions.shape)
    ax.set_xticklabels([''] + input_sentence.split(' ') , rotation=90 ,fontsize = fsz)
    ax.set_yticklabels([''] + output_words.split(' ') ,fontsize=fsz)
    ax.set_title("Input words" ,fontsize=fsz ,color='red')
    # ax.set_xlabel("input words)
    ax.set_ylabel("predicted words" ,fontsize=fsz ,color='green')
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


def calculate_attention_batch(data_loader ,loaded_model ,vocab_obj,indx_batch=0,word=None,multiple_references=True,device=None):
    loaded_model.eval()
    if word: indx_batch=-1
    for idx, batch in enumerate(data_loader):
        if idx == indx_batch: break
    #bx,by,lens= batch
    # Calculate predictions
    src = batch[0].to(device).permute(1, 0, 2)
    src = torch.as_tensor(src, dtype=torch.float32)
    # shape (batch_size,src_len,flatten joint dim = 21*3)
    trg = batch[1].to(device).permute(1, 0) if not multiple_references else \
        pad_sequence([torch.as_tensor(refs[0]) for refs in batch[1]], batch_first=False, padding_value=0).to(device)
    src_lens = batch[2]  # (batch_size,)
    trg_lens = batch[3]
    hidden_size = 64
    init_hidden = torch.zeros((2, src.size(1), hidden_size)).to(device)
    output_pose = loaded_model(src, trg, init_hidden, teacher_force_ratio=0, src_lens=src_lens)

    _dec_numeric_sentence = vocab_obj.decode_numeric_sentence
    # Decode Predictions
    bleu_score, output_predictions, target_sentences = decode_predictions_and_compute_bleu_score(output_pose, batch[1] if multiple_references else trg,
                                                                 vocab_obj,num_grams=4, batch_first=False,multiple_references=multiple_references)

    print("BLEU-4 SCORE BATCH %.3f" % bleu_score)
    #print("input\n", target_sentences, "\noutput\n", output_predictions)
    att_w = loaded_model.attention_weights.cpu().detach().numpy()

    return  att_w,target_sentences,output_predictions,src_lens,src,loaded_model.attention_positions.cpu().detach().numpy()


def visualize_attention_m2l(target_sentence, output_words, attentions ,fig ,i ,fsz=20,start_pad=-1):
    # Set up figure with colorbar
    # fig = plt.figure()
    print("words len : target sentence ", len(target_sentence.split(' ')))
    print("words len : output", len(output_words.split(' ')),output_words)
    print("attention shape", attentions.shape)
    seq_len_pose = attentions.shape[1]

    #fig.set_facecolor("white")
    predictions = output_words.split(' ') # + ['<eos>']
    # Set up axes
    ax = fig.add_subplot(i)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(ticker.FixedLocator(list(range(len(predictions)))))
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(start_pad))))

    ax.set_yticklabels(predictions, rotation=0, fontsize=fsz)

    #ax.set_xticklabels([''] + [k for k in range(seq_len_pose)] , rotation=90 ,fontsize = fsz)
    #Show label at every tick

    ax.set_xlabel(f"Index pose frames start_pad : {start_pad}" ,fontsize=fsz)
    ax.set_ylabel("Predicted words" ,fontsize=fsz ,color='green')
    ax.set_title(f"Target words {target_sentence}" ,fontsize=fsz ,color='red')

    cax = ax.matshow(attentions[:len(predictions),:start_pad])  # , cmap='bone')
    #fig.colorbar(cax,orientation = "horizontal" )
    fig.tight_layout()
    plt.show()
    #ax.set_yticklabels(predictions,fontsize=fsz)

    #ax.set_xticklabels([''] + [f"frame_{k}" for k in range(seq_len_pose)],fontsize=fsz)


