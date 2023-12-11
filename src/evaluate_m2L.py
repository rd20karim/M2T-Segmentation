import os
import sys
# sys.path.extend([os.getcwd()])
# print(sys.path)
import argparse
import logging
import torch
import yaml
from datasets.loader import build_data
from tune_train import run_batch
from bleu_from_csv import read_pred_refs,calculate_bleu
global min_freq,batch_size,hidden_size,embedding_dim
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("out_eval_m2l.txt"),
        logging.StreamHandler()
    ]
)

def load_model_config(args= None,device=None,load_data=True,input_dim=None):
    multiple_references = args.multiple_references
    input_angles = True if args.input_type != "cartesian" else False
    encoder_type = args.encoder_type; mask = args.mask
    path_weights = args.path; attention_type = args.attention_type
    bidirectional = True if "Bi" in encoder_type else False
    hidden_size = args.hidden_size; embedding_dim = args.embedding_dim
    min_freq = args.min_freq; batch_size = args.batch_size


    project_path = r"C:\Users\karim\PycharmProjects\SemMotion"
    aug_path = r"C:\Users\karim\PycharmProjects\HumanML3D"

    if "kit" in args.dataset_name:
        # -------------KIT IMPORTS------------------
        from architectures.crnns import seq2seq
        if "2016" in args.dataset_name:
            from datasets.kit_m2t_dataset_2016 import dataset_class
            path_txt = project_path+"\sentences_corrections_origin.csv"  # os.getcwd()+"\sentences_corrections_origin.csv"
            path_motion = project_path+"\datasets\kitmld_anglejoint_2016_30s_final_cartesian.npz"

        else: # ------------ [Augmented-KIT] ------------
            from datasets.kit_m2t_dataset import dataset_class
            path_txt = project_path+"\datasets\sentences_corrections.csv"
            path_motion = aug_path+"\kit_with_splits_2023.npz"

            # -----------H3D IMPORTS---------------------
    elif args.dataset_name=="h3D":
        from architectures.crnns_H3D import seq2seq
        from datasets.h3d_m2t_dataset_ import dataset_class
        path_txt = aug_path+"\sentences_corrections_h3d.csv"
        path_motion = aug_path+"\\all_humanML3D.npz"

    if load_data :
        train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class, min_freq= min_freq,
                                                                          path = path_motion,
                                                                          train_batch_size=batch_size,
                                                                          test_batch_size=batch_size,
                                                                          return_lengths=True, path_txt=path_txt,
                                                                          return_trg_len=True, joint_angles=input_angles,
                                                                          multiple_references=multiple_references,
                                                                          random_state=args.random_state)
        input_dim = train_data_loader.dataset.lang.vocab_size_unk
    else :
        train_data_loader, val_data_loader, test_data_loader = None,None,None

    logging.info("VOCAB SIZE  = %d "  % (input_dim))
    loaded_model = seq2seq(input_dim, hidden_size, embedding_dim, num_layers=1, device=device, bidirectional=bidirectional,
            attention=attention_type, mask=mask, joint_angles=input_angles, encoder_type=encoder_type,
            beam_size=args.beam_size, D=args.D, scale=args.scale, hidden_dim=args.hidden_dim).to(device)

    weights = torch.load(path_weights,map_location=torch.device(args.device))
    new_dict = weights[0]

    loaded_model.load_state_dict(new_dict)
    return loaded_model,train_data_loader,test_data_loader,val_data_loader

def evaluate(loaded_model,data_loader,mode,multiple_references=False,input_type ="angles",name_file=None,beam_size=1):
    loaded_model.eval()
    epoch_loss = 0
    output_per_batch,target_per_batch = [],[]
    name_file = f"{input_type}_{name_file}_"+args.dataset_name+"_"+args.encoder_type
    BLEU_scores = []
    with open(name_file + ".csv", mode="w") as _:pass
    logging.info(f"Compute BLEU scores per batch and write predictions/refs to {name_file}")
    loaded_model.eval()
    if beam_size==1:
        for i, batch in enumerate(data_loader):
            loss_b,bleu_score_4,pred,refs = run_batch(model=loaded_model,batch=batch,data_loader=data_loader,mode=mode,teacher_force_ratio=0,
                                                          device=device,multiple_references=multiple_references,BETA=args.beta)
            BLEU_scores.append(bleu_score_4)
            # write predictions and targets for every batch
            with open(name_file + ".csv", mode="a") as f:
                for p, t in zip(pred, refs):
                    f.writelines(("%s" + ",%s" * len(t) + "\n") % ((" ".join(p).replace("\n", ""),) + tuple(" ".join(k).replace("\n", "") for k in t)))
            logging.info("Loss/test_batch %d --> %.3f  BLEU score_batch %.3f" % (i, loss_b, bleu_score_4))
            epoch_loss += loss_b.item()
        loss = epoch_loss / len(data_loader)
        BLEU_score = sum(BLEU_scores) / len(BLEU_scores)
        logging.info(f"LOSS {mode} %.3f BLEU_4 score %.3f" % (loss, BLEU_score))
        """ Beam search evaluating """
    else:
        first_bleus = []
        for i, batch in enumerate(data_loader):
            bleu_score_beam, predicted_sentences, refs = run_batch(model=loaded_model, batch=batch, data_loader=data_loader, mode=mode, teacher_force_ratio=0,
                                                                device=device, multiple_references=multiple_references, beam_size=beam_size, BETA=args.beta)
            for bm, sc in enumerate(bleu_score_beam):
                logging.info(f"BLEU_4 score beam {bm} --> %.3f" % (sc,))
                if bm==0 : first_bleus.append(sc)

            print(sum(first_bleus)/len(first_bleus))
    return output_per_batch,target_per_batch

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,help="Path of model weights not used if the config file is passed")
    parser.add_argument("--dataset_name",type=str,default="kit2016",choices=["h3D","kit","kit2016"])
    parser.add_argument("--config",type=str,default="./configs/local_rec/MLP.yaml")
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--input_type",type=str,default="cartesian",choices=["cartesian","angles"])
    parser.add_argument("--multiple_references",type=str,default="True",choices=["True","False"],help="Specify evaluation mode use flattened references or all at one")
    parser.add_argument("--encoder_type",type=str,default="MLP",choices=["GRU","BiGRU","MLP","deep-MLP"])
    parser.add_argument("--attention_type",type=str,default="local recurrent",choices=["local recurrent","local","soft"])
    parser.add_argument("--mask",type=str,default="True",choices=["True","False"])
    parser.add_argument("--name_file", type=str, default="", help="File name of predicted sentences")
    parser.add_argument("--subset", type=str, default="test", help="Subset on which evaluating the data",choices=["test", "val","train"])
    parser.add_argument("--hidden_dim", type=int, default=128, help='hidden dimension of the encoder layers for the MLP')
    parser.add_argument("--hidden_size", type=int, default=64, help='hidden size of the decoder and encoder output dimension')
    parser.add_argument("--embedding_dim", type=int, default=64, help='embedding dimension of words')
    parser.add_argument("--min_freq", type=int, default=3, help='minimum word frequency to keep in the dataset')
    parser.add_argument("--beam_size", type=int, default=1, help='beam search width')
    parser.add_argument("--use_unknown_token", type=str, default=True, help='To use or not the unknown token for evaluation')
    parser.add_argument("--beta",type=int,default=0,help="Beta normalizing loss factor")
    parser.add_argument("--D",type=int,default=5,help="Half window length")
    parser.add_argument("--random_state",type=int,default=11,help="random_state")
    parser.add_argument("--scale",default=1.,help="specify a float value or set True for automatic selection")
    parser.add_argument("--batch_size", type=int, default=512, help='Batch size should be >= to length of data for Corpus BLEU score')
    args = parser.parse_args()

    # Update args parser
    with open(args.config,'r') as f:
        choices = yaml.load(f,Loader=yaml.Loader)
    default_arg = vars(args)
    parser.set_defaults(**choices)
    args = parser.parse_args()

    device = torch.device(args.device)
    logging.info(f"INPUT {args.input_type} Encoder {args.encoder_type}")
    loaded_model, train_data_loader, test_data_loader, val_data_loader = load_model_config(device=device,args=args)
    data_loader = locals()[args.subset+"_data_loader"]

    if "loc" in args.attention_type : logging.info(f"{args.attention_type} [ Mask : {args.mask}, D: {args.D} ]")
    logging.info(f"Checkpoint on {args.subset} set")

    output_per_batch, target_per_batch = evaluate(loaded_model=loaded_model, data_loader=data_loader,multiple_references=args.multiple_references,
                                                  input_type=args.input_type,name_file=args.name_file,beam_size=args.beam_size,mode="eval")
