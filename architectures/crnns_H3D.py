import logging

import numpy as np
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from .decode_beam import beam_decode
class Encoder(nn.Module):
    def __init__(self, input_dim,hidden_size, embedding_dim, num_layers=1, bidirectional=True,type='pose',
                 dropout=None,joint_angles=True,encoder_type="GRU",hidden_dim=128,n_joints=21):
        super().__init__()
        self.type = type
        self.dropout = nn.Dropout(dropout)
        self.encoder_type = encoder_type
        self.norm_f = lambda x,norm:(x-x.mean(-1).unsqueeze(-1))/(x.std(-1).unsqueeze(-1)+1e-5) if norm else x
        dims =  n_joints*3
        if "GRU" in encoder_type :
            self.rnns = nn.GRU(dims, hidden_size, bidirectional=bidirectional, num_layers=num_layers)
            self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=None)
        else:
            self.rnns = nn.GRU(dims, hidden_size, bidirectional=False, num_layers=1)
            self.fc1 = nn.Linear(dims, hidden_dim)
            if "deep" in encoder_type:
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, hidden_dim)
                self.fc4 = nn.Linear(hidden_dim, hidden_size)
            else:
                self.fc2 = nn.Linear(hidden_dim, hidden_size)
    def forward(self, x, previous_hidden,src_lens=None,norm=True):
        if "GRU" in self.encoder_type :
            if self.type!='pose' : x = self.embedding(x); x = self.dropout(x)
            x = pack_padded_sequence(x, src_lens, enforce_sorted=False)  # src #
            outputs, hidden_states = self.rnns(x, previous_hidden)
            return outputs, hidden_states
        else:
            x = self.dropout(torch.tanh(self.norm_f(self.fc1(x),norm)))
            x = self.dropout(torch.tanh(self.norm_f(self.fc2(x),norm)))
            if "deep" in self.encoder_type:
                x = self.dropout(torch.tanh(self.norm_f(self.fc3(x),norm)))
                x = self.dropout(torch.tanh(self.norm_f(self.fc4(x),norm)))
            return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim, batch_size=32, attention="bahadanau", enc_bidirectional=True,
                 num_layers=1, device=torch.device("cpu"),dropout=None,mask=None,beam_search=False,D=5,scale=1.):
        super().__init__()
        self.mask = mask
        self.device = device
        self.beam_search = beam_search
        self.batch_size = batch_size
        self.D = D
        self.dropout = nn.Dropout(dropout)
        k = 2 if enc_bidirectional else 1
        self.dec_hidden_size = k * num_layers * hidden_size
        self.output_dim = input_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=None)
        # TODO WAS CHANGED
        self.rnns = nn.GRU(input_size=embedding_dim + self.dec_hidden_size ,num_layers=num_layers,
                           hidden_size=self.dec_hidden_size , bidirectional=False)
        self.norm_f = lambda x,norm=True:(x-x.mean(-1).unsqueeze(-1))/(x.std(-1).unsqueeze(-1)+1e-5) if norm else x
        try : scale = float(scale)
        except: scale = bool(scale)

        self.scale = float(scale) if type(scale)==float else 1/np.sqrt(self.dec_hidden_size)
        logging.info("Scaling "+ str(self.scale))
        self.fc = nn.Linear(in_features= embedding_dim +  self.dec_hidden_size + self.dec_hidden_size, out_features=self.output_dim) # We feed the concatenation of context and previous word and GRU output
        self.attention_type = attention
        self.weights_contribution = None
        logging.info(f"Applying {self.attention_type} attention")

        self.enc_bidirec  = enc_bidirectional
        if "local" in self.attention_type:
            self.dec_hidden_layer = nn.Linear(self.dec_hidden_size, self.dec_hidden_size)
            self.position_layer = nn.Linear(self.dec_hidden_size, self.dec_hidden_size)  # Ua
            self.window_layer = nn.Linear(self.dec_hidden_size, 1)
            self.proj_vector = nn.Linear(self.dec_hidden_size, 1, bias=False)  # va
            self.sigma_proj = nn.Linear(self.dec_hidden_size, 1, bias=False)  # va

        else:
            self.dec_hidden_layer = nn.Linear(self.dec_hidden_size, self.dec_hidden_size)
            self.enc_hidden_layer = nn.Linear(self.dec_hidden_size, self.dec_hidden_size)  # Ua
            self.proj_vector = nn.Linear(self.dec_hidden_size, 1, bias=False)  # va


    def calculate_attention_weight(self, previous_hidden_dec,enc_masks=None,enc_outputs=-1):
        #TODO review this implementation concatenate multiple output layers
        previous_hidden_dec =  previous_hidden_dec[-1,:,:]
        #shape previous_hidden_dec [N,Hout*D*num_layers]
        src_len = len(self.wh)
        self.dw = self.dec_hidden_layer(previous_hidden_dec)  # Wa*s_(i-1)
        self.act = torch.empty(size=(src_len, self.batch_size, self.dec_hidden_size), device=self.device)
        # CALCULATE ENERGY COEFFICIENT FOR EACH INPUT SRC_Len
        if "local" in self.attention_type:
            # dot-product (general)
            self.energy_align_model = torch.tanh(torch.bmm(self.dw.unsqueeze(1), self.wh.permute(1, 2, 0))).squeeze(1).permute(1,0)
            #infer src_lens
            src_lens = (torch.argmin(enc_masks.T, dim=1)).to(self.device)
            src_lens[src_lens == 0] = src_len
            # self.energy_align_model shape : [src_len,batch_size,1]
            # Mask coefficient of padding
            src_lens[src_lens == 0] = src_len
            self.scores = self.energy_align_model.masked_fill(enc_masks.to(self.device) == 0, float('-inf')).unsqueeze(2)
            self.att_w = torch.softmax(self.scores, dim=0) # softmax(eij) = alpha_ij
        else:
            # "Bahadanau" attention  : va.T * tanh(Wa*s_(i-1)+Ua*h_j) = e_ij
            # act shape  : [src_len,batch_size,hidden_size]
            self.act = torch.tanh(self.dw.unsqueeze(0) + self.wh)
            self.energy_align_model = self.proj_vector(self.act).squeeze(2)
            # mask coefficient of padding --> softmax == 0
            self.scores = self.energy_align_model.masked_fill(enc_masks.to(self.device) == 0, float('-inf')).unsqueeze(2)
            self.att_w = torch.softmax(self.scores, dim=0)  # softmax(eij) = alpha_ij
            # self.att_w : [src_len,batch_size,1]
        return self.att_w

    def forward(self, x, previous_hidden, encoder_outputs,enc_masks=None,epsilon=0):
        # x shape [1,batch_size]
        x = self.embedding(x)
        x = self.dropout(x)
        self.batch_size = x.size(1)  # it's variable in the final batch
        src_len = encoder_outputs.size(0)
        src_lens = (torch.argmin(enc_masks.T, dim=1)).to(self.device)
        src_lens[src_lens == 0] = src_len
        # x shape [1,batch_size,embedding_dim]
        if self.attention_type:
           # encoder_outputs shape : [src_len,batch_size,k*num_layers*encoder_hidden_size] (k=2 if bidirectional)
            src_len = encoder_outputs.size(0)
            self.wh = encoder_outputs if "local" in self.attention_type  else self.enc_hidden_layer(encoder_outputs)
            self.weights_att = self.calculate_attention_weight(previous_hidden,enc_masks=enc_masks,enc_outputs=encoder_outputs)
            if "local" in self.attention_type:
                # Alignment position p_t HERE DEPEND ONLY ON DECODER HIDDEN STATE
                self.pt = self.pt.reshape((self.batch_size,)) if self.pt.ndim == 1 else self.pt[0, :, 0]

                if self.all_pt.shape[1]!=0:
                    f_ht = torch.sigmoid(self.proj_vector(torch.tanh(
                                                self.norm_f(self.position_layer(previous_hidden),norm=False)))
                                                            )
                    if not self.beam_search: f_ht = f_ht.squeeze(2).squeeze(0)
                    if "rec" in self.attention_type:
                        self.pt = self.pt + epsilon + (src_lens - 1 - self.pt - epsilon) * f_ht
                    else:
                        self.pt =(src_lens - 1) * f_ht

                self.all_pt = torch.cat([self.all_pt, self.pt.reshape((self.batch_size, 1))], axis=1)

                # self.pt [batch_size,]
                # TODO study D parameter empirically
                D = self.D * torch.ones((len(src_lens), 1), device=self.device)
                s = torch.arange(0, src_len, device=self.device).reshape(src_len, 1, 1).expand(src_len, len(src_lens),1)
                sigma = D.reshape(1, len(src_lens), 1).expand(s.size()) / 2
                self.pt = self.pt.reshape(1, len(self.pt), 1).expand(s.size())
                window = torch.exp(-(self.pt - s) ** 2 / (2 * sigma ** 2))
                if self.mask:
                    # Calculate the maximum and minimum positions of the window for each batch
                    pmax = torch.floor(self.pt[0, :, 0] + D[:, 0]).long()  # [batch_size]
                    pmin = self.pt[0, :, 0] - D[:, 0]  # [batch_size]
                    pmin[pmin < 0] = 0  # Clip negative values to 0
                    pmin = torch.floor(pmin).long()
                    # Create a mask where positions outside the window are 0
                    mask = (torch.arange(window.shape[0], device=self.device)[:, None] < pmax) & \
                           (torch.arange(window.shape[0], device=self.device)[:, None] >= pmin)
                    mask = mask.unsqueeze(-1).float()  # [seq_len, batch_size, 1]
                    # Truncated window
                    window = window * mask  #  apply the mask to the input window

                self.weights_att = self.weights_att * window

            # self.weights_att shape : [src_len,batch_size,1]
            self.context_vector = torch.bmm(self.weights_att.permute(1, 2, 0),
                                            encoder_outputs.permute(1, 0, 2))  # c_i = SUM_j(alpha_ij*hj)
            # self.contex_vector shape : [batch_size, 1,k*num_layers*encoder_hidden_size]

            self.dec_input = torch.cat((x, self.context_vector.permute(1, 0, 2)), dim=2)

        # self.dec_input shape : [1,batch_size, k*num_layers*encoder_hidden_size + embedding_dim]
        dec_outputs, dec_final_hidden = self.rnns(self.dec_input, previous_hidden)
        f_g = (x,dec_final_hidden[-1].unsqueeze(0), self.context_vector.permute(1, 0, 2))
        new_input = torch.cat(f_g, dim=2)
        # todo Reactive this part to show weight contributions of language and motion
        #self.contribution(x,f_g,dec_final_hidden)
        z = self.fc(new_input)
        # x shape : [batch_size, output_dim]

        return z, dec_final_hidden

    def contribution(self,x,f_g,dec_final_hidden):
        if "local" in self.attention_type:
            Lang = (x,torch.zeros_like(f_g[1]),torch.zeros_like(f_g[2]))
            Motion = (x,torch.zeros_like(f_g[1]),self.context_vector.permute(1, 0, 2))
            mix_motion_lang = (torch.zeros_like(f_g[0]),dec_final_hidden[-1].unsqueeze(0),torch.zeros_like(f_g[2]))
            contributions = torch.empty((self.batch_size,3))
            for i, f_i in enumerate([Lang, Motion, mix_motion_lang]):
                spec_f = torch.cat(f_i, dim=2)
                contributions[:,i]  = torch.max(self.fc(spec_f), dim=2).values
            if self.weights_contribution is None : self.weights_contribution = torch.empty((self.batch_size, 0, 3))
            self.weights_contribution = torch.cat([self.weights_contribution,contributions.unsqueeze(1)] ,dim=1)
           # self.weights_contribution shape : (batch_size, maximal sentence length, 3)


class seq2seq(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim, num_layers=1,device=torch.device('cpu'),dropout=0,beam_size=1,
                 joint_angles=False,bidirectional=True,attention="bahadanau",mask=True,encoder_type="GRU",hidden_dim=128,D=5,scale=1.):
        super(seq2seq, self).__init__()
        self.device = device
        self.output_dim = input_dim # vocab_size
        self.hidden_size = hidden_size
        self.encoder_type=encoder_type
        self.enc_pose = Encoder(input_dim, hidden_size, embedding_dim, num_layers=num_layers, dropout=dropout,
                                joint_angles=joint_angles, bidirectional=bidirectional,encoder_type=encoder_type,hidden_dim=hidden_dim,n_joints=22)
        self.beam_size = beam_size
        self.dec = Decoder(input_dim, hidden_size, embedding_dim, num_layers=num_layers, device=device,
                           dropout=dropout,enc_bidirectional=bidirectional,attention=attention,mask=mask,
                           beam_search=True if self.beam_size>1 else False,D=D,scale=scale)
        self.attention_type = attention
        self.num_layers = num_layers


    def forward(self, x, y, init_hidden, teacher_force_ratio=0,src_lens=None,beam_size=1):
        # TODO add case no src_lens
        self.src_lens = src_lens # for packed sequence
        enc_masks = torch.zeros(x.shape[:2]) # (seq_len,batch_size)
        for i,l in enumerate(src_lens): enc_masks[:l,i]=1
        D = 2 if self.enc_pose.rnns.bidirectional else 1
        init_hidden = torch.zeros((D*self.num_layers, x.size(1), self.hidden_size),device=self.device)
        if "GRU" in self.encoder_type:
            enc_outputs, enc_final_hidden =  self.enc_pose(x, init_hidden,src_lens=self.src_lens)
            previous_dec_hidden = torch.cat([enc_final_hidden[-k, :, :] for k in range(len(enc_final_hidden), 0, -1)],dim=1).unsqueeze(0)
            enc_outputs, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(enc_outputs)
        else:
            enc_outputs= self.enc_pose(x, init_hidden,src_lens=self.src_lens)
            previous_dec_hidden = torch.zeros((1,x.size(1), self.hidden_size),device=self.device)
        dec_pred_output = []
        dec_pred_output.append(torch.ones((1, y.size(1)), dtype=torch.int, device=self.device))  # first tokens : <sos> index : 1
        trg_len = y.size(0)
        output_list = []
        self.attention_weights_list = []
        # todo re-do find index of padding here is 0

        self.dec.pt = torch.zeros((len(src_lens),), device=self.device)

        self.dec.all_pt = torch.empty((x.size(1),0),device=self.device)
        self.beam_size = beam_size
        if self.beam_size==1:
            for j in range(trg_len-1):
                thr = random.random()
                y_s = y[j].unsqueeze(0) if thr < teacher_force_ratio else dec_pred_output[j]
                dec_output, previous_dec_hidden = self.dec(y_s, previous_dec_hidden, enc_outputs,enc_masks)
                self.dec_output = dec_output
                self.attention_weights_list.append(self.dec.weights_att)
                output_list.append(dec_output)
                self.output_list = output_list
                dec_next_input = torch.argmax(torch.softmax(dec_output, dim=2), dim=2)  # (1,batch_size)
                dec_pred_output.append(dec_next_input)
                self.dec_next_input = dec_next_input

            dec_pred_output = torch.cat(dec_pred_output, dim=0)
            # For evaluation
            self.dec_pred_output = dec_pred_output
            self.attention_weights = torch.cat(self.attention_weights_list, dim=2)
            self.target_and_prediction = [y, dec_pred_output]
            outputs_logits = torch.cat(output_list, dim=0)
            #TODO To combine special for MLP line bellow
            self.attention_positions = torch.round(self.dec.all_pt) if "local" in self.attention_type else torch.argmax(self.attention_weights,dim=0)
            # outputs shape : (trg_len,batch_size, output_dim) output_dim --> logits
            return outputs_logits
        else:
            self.x = x
            decoded_batch, attention_batch, att_pos_batch = beam_decode(self, y, previous_dec_hidden, enc_outputs)
            # fetch best attentions
            self.attention_positions = att_pos_batch
            self.attention_weights = attention_batch
            return decoded_batch
