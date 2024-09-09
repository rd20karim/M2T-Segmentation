import sys
import os
sys.path.extend(os.getcwd())
import argparse
from datasets.visualization import decode_predictions_and_compute_bleu_score
import  logging
from datasets.loader import build_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import  yaml
import torchtext


def run_batch(model, batch, data_loader, mode, teacher_force_ratio, device=None, optimizer=None,
              multiple_references=None, BETA=None, beam_size=1,epoch=0):
    epoch_loss = 0
    TRG_PAD_IDX = data_loader.dataset.lang.token_to_idx["<pad>"]
    src = batch[0].to(device).permute(1, 0, 2)
    src = torch.as_tensor(src, dtype=torch.float32)
    # shape (batch_size,src_len,flatten joint dim = 21*3)
    trg = batch[1].to(device).permute(1, 0) if not multiple_references else \
        pad_sequence([torch.as_tensor(refs[0]) for refs in batch[1]], batch_first=False, padding_value=0).to(device)

    src_lens = batch[2]
    trg_lens = batch[3]
    init_hidden = torch.zeros((2, src.size(1), 64)).to(device)
    num_grams = 4
    vocab_obj = data_loader.dataset.lang
    if beam_size==1:
        if "test" in mode : logging.info("START Greedy SEARCH ")
        ## Run model
        output_pose = model(src, trg, init_hidden, teacher_force_ratio=teacher_force_ratio, src_lens=src_lens)

        bleu_score, pred, refs = decode_predictions_and_compute_bleu_score(output_pose.squeeze(0), batch[1] if multiple_references else trg,
                                                                           vocab_obj,num_grams=num_grams, batch_first=False,
                                                                           multiple_references=multiple_references)
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX, reduction='mean')
        loss = criterion(output_pose.permute(1, 2, 0), trg[1:, :].permute(1, 0))

        logging.info(f"loss {loss.item()}")

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        epoch_loss += loss.item()
        return loss, bleu_score, pred, refs
    # Only for evaluation
    else:
        logging.info("START BEAM SEARCH")
        decoded_preds = model(src, trg, init_hidden, teacher_force_ratio=0, src_lens=src_lens)
        predicted_sentences = []
        _dec_numeric_sentence = vocab_obj.decode_numeric_sentence
        for hyps in decoded_preds:
            predicted_sentences += [
                [_dec_numeric_sentence(beam_path, remove_sos_eos=True).split(
                    " ") for beam_path in hyps]]
        logging.info("Write beam predictions ...")
        filename = f"result_beamsize_{beam_size}_.txt"
        with open(filename, "w") as g:
            for m in predicted_sentences:
                g.writelines([" ".join(k) + "," for k in m] + ["\n"])
        Yrefs = batch[1] if multiple_references else trg
        ref_sentences = [[_dec_numeric_sentence(ref, remove_sos_eos=True).split(" ") for ref in refs] for refs in Yrefs]
        bleu_score_beam = [torchtext.data.metrics.bleu_score(
            candidate_corpus=[m[k] if len(m) >= k + 1 else m[-1] for m in predicted_sentences],
            references_corpus=ref_sentences,
            max_n=num_grams, weights=[1 / num_grams] * num_grams) for k in range(beam_size)]
        return bleu_score_beam, predicted_sentences, Yrefs

def train_m2l(config,data=None): 
    if "kit" in args.dataset_name:
        # -------------KIT IMPORTS------------------
        from architectures.crnns import seq2seq
        if "2016" in args.dataset_name:
            from datasets.kit_m2t_dataset_2016 import dataset_class
            path_txt = r"C:\Users\karim\PycharmProjects\SemMotion\sentences_corrections_origin.csv"  # os.getcwd()+"\sentences_corrections_origin.csv"
            path_motion = r"C:\Users\karim\PycharmProjects\SemMotion\datasets\kitmld_anglejoint_2016_30s_final_cartesian.npz"

        else: # [Augmented-KIT]
            from datasets.kit_m2t_dataset import dataset_class
            path_txt = None
            path_motion = r"C:\Users\karim\PycharmProjects\HumanML3D\kit_with_splits_2023.npz"

    elif args.dataset_name=="h3D":
        # -----------H3D IMPORTS---------------------
        from architectures.crnns_H3D import seq2seq
        from datasets.h3d_m2t_dataset_ import dataset_class
        # TODO CHANGE THIS PATH
        path_txt = r"C:\Users\karim\PycharmProjects\HumanML3D\sentences_corrections_h3d.csv"
        path_motion = r"C:\Users\karim\PycharmProjects\HumanML3D\all_humanML3D.npz"

    train_data_loader, val_data_loader, test_data_loader = build_data(dataset_class=dataset_class, min_freq=config["min_freq"],
                                                                      path=path_motion,
                                                                      train_batch_size=config["batch_size"],
                                                                      test_batch_size=config["batch_size"],
                                                                      return_lengths=True, path_txt=path_txt,
                                                                      # r"{}".format(path_txt)
                                                                      return_trg_len=True, joint_angles= False,
                                                                      multiple_references=False)


    "Define Model"
    input_dim = train_data_loader.dataset.lang.vocab_size_unk
    bidirectional = True if "Bi" in config["encoder_type"] else False
    model = seq2seq(input_dim, config["hidden_size"], config["embedding_dim"], num_layers=config["num_layers"],
                    device=config["device"],dropout =config["rate_dropout"] ,bidirectional=bidirectional,
                    attention=config["attention_type"],mask=config["mask"],joint_angles=True if config["input_type"]!="cartesian" else False,
                    encoder_type=config["encoder_type"],hidden_dim=config["hidden_dim"],D=config["D"],scale=config["scale"])

    if args.resume_epoch:
        logging.info("Resuming final model")
        path_weights = r"C:\Users\karim\ray_results\checkpoint"
        model.load_state_dict(torch.load(path_weights)[0])

    """ Parallelization    """
    gpu_ids = [0, 1]
    primary_gpu_id = gpu_ids[0]
    model = model.to(config["device"])
    logging.info(f"Model Architecture {model}")
    n_epochs = config["n_epochs"]

    logging.info("************ START TRAINING ************")
    start = args.resume_epoch
    optimizer = optim.Adam(model.parameters(),lr=config['lr'])

    for epoch in range(start,n_epochs):
        # TRAIN
        model.train()
        teacher_force_ratio = config["teacher_force_ratio"]

        epoch_loss = 0
        BLEU_scores = []
        mode  = "train"
        for i, batch in enumerate(train_data_loader):
            loss_train_b, bleu_score,_,_ = run_batch(model,batch,train_data_loader, mode=mode,optimizer=optimizer,
                                                     teacher_force_ratio=teacher_force_ratio,device=config["device"],BETA=config["beta"],epoch=epoch)
            BLEU_scores += [bleu_score]
            loss_train_b = loss_train_b.item()
            epoch_loss += loss_train_b
            logging.info(f"Loss/{mode}_batch %d --> %.3f BLEU score_batch %.3f" % (i, loss_train_b, bleu_score))

        loss_train = epoch_loss / len(train_data_loader)
        BLEU_score_train = sum(BLEU_scores) / len(BLEU_scores)
        logging.info(f"\nEpoch %d Train Loss --> %.3f BLEU_train score  %.3f\n" % (epoch, loss_train, BLEU_score_train))
        # EVALUATE
        # TODO ADDED SEPARATELY THIS CASE
        evaluate = True
        if evaluate:
            mode = "val"
            model.eval()
            epoch_loss = 0
            BLEU_scores = []
            for i, batch in enumerate(val_data_loader):
                loss_val_b, bleu_score, _, _ = run_batch(model, batch, val_data_loader, mode=mode,optimizer=optimizer,
                                                         teacher_force_ratio=teacher_force_ratio,device=config["device"],BETA=config["beta"],epoch=epoch)
                BLEU_scores += [bleu_score]
                loss_val_b = loss_val_b.item()
                epoch_loss += loss_val_b
                logging.info(f"Loss/{mode}_batch %d --> %.3f BLEU score_batch %.3f" % (i, loss_val_b, bleu_score))

            loss_val = epoch_loss / len(val_data_loader)
            BLEU_score_val = sum(BLEU_scores) / len(BLEU_scores)
            logging.info("LOSS VAL %.3f BLEU score %.3f" % (loss_val, BLEU_score_val))
            logging.info(f"\nEpoch  %d LOSS VAL %.3f  BLEU_val score %.3f" % (epoch, loss_val, BLEU_score_val))
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss_val = loss_val, bleu_val=BLEU_score_val,
                        loss_train = loss_train, bleu_train=BLEU_score_train,epoch=epoch)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path",type=str,default=".",help="Path where to save checkpoints")
    parser.add_argument("--dataset_name",type=str,default="kit2016",choices=["h3D","kit","kit2016"])
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--config",type=str,default="./configs/MLP.yaml")
    parser.add_argument("--input_type",type=str,default="cartesian",choices=["cartesian","angles"])
    parser.add_argument("--multiple_references",type=bool,default=False,help="Specify evaluation mode use flattened references or all at one")
    parser.add_argument("--encoder_type",type=str,default="MLP",choices=["GRU","BiGRU","MLP","deep-MLP"])
    parser.add_argument("--attention_type",type=str,default="local_recurrent",choices=["local_recurrent","local","soft"])
    parser.add_argument("--mask",type=bool,default=True,choices=[True,False])
    parser.add_argument("--experience_suffix_name",type=str,default="_exp0",help='Run name')
    parser.add_argument("--epoch",type=int,default=1000,help='Number of epoch')
    parser.add_argument("--save_checkpoint",type=bool,default=True,help="save checkpoint at each end")
    parser.add_argument("--beta",type=int,default=0,help="Beta normalizing loss factor")
    parser.add_argument("--random_state",type=int,default=11,help="random_state")
    parser.add_argument("--scale",default=1.,help="specify a float value or set True for automatic selection")
    parser.add_argument("--resume_epoch",type=int,default=0,help="epoch number from which resume the training")
    parser.add_argument("--lr",type=int,default=0.001,help="epoch number from which resume the training")

    args = parser.parse_args()

    with open(args.config,'r') as f:
        choices = yaml.load(f,Loader=yaml.Loader)
    default_arg = vars(args)
    parser.set_defaults(**choices)
    args = parser.parse_args()

    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    import ray

    config = {
                "hidden_size":tune.grid_search(choices["hidden_size"]),
                "embedding_dim":tune.grid_search(choices['embedding_dim']),
                "hidden_dim": tune.grid_search(choices["hidden_dim"]),
                "lr":tune.grid_search(choices['lr']),
                "beta":0,
                "batch_size": tune.grid_search(choices['batch_size']),
                "num_layers":1,
                "min_freq": 3,
                "teacher_force_ratio" :tune.grid_search(choices["teacher_force_ratio"]),
                'device': torch.device(args.device),
                "rate_dropout": 0.5,
                "mask":tune.grid_search(choices["mask"]),
                "n_epochs":args.epoch,
                "D": tune.grid_search(choices["D"]),
                "encoder_type": tune.grid_search(choices["encoder_type"]),
                "sheduler":tune.grid_search(["adam"]),
                "random_state":args.random_state,
                "scale": args.scale,
                "attention_type":tune.grid_search(choices["attention_type"]),
                "input_type" : choices["input_type"]
            }

    gpus_per_trial = 1
    num_samples  = 1
    max_num_epochs = config["n_epochs"]

    # ...
    scheduler = ASHAScheduler(metric="bleu_val", mode="max", max_t=max_num_epochs, grace_period=max_num_epochs , reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss_val", "bleu_val", "loss_train", "bleu_train", "training_iteration"])
    logging.info(f"training on device :{config['device']}" )

    ray.shutdown()
    from ray.tune import Stopper
    
    ray.init()
    result = tune.run(train_m2l,
        resources_per_trial={"gpu": gpus_per_trial},
        config= config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=False,
        metric=None,
        mode=None,
        name= args.experience_suffix_name, #f'{args.input_type}_{args.encoder_type}_{args.attention_type}'+args.experience_suffix_name,
       storage_path= r"/Users/karim/ray_results/")
    ray.shutdown()