
import operator
import torch
from queue import PriorityQueue
import logging
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length,att_weights,att_position=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        :param att_weights:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.att_weights = att_weights
        self.att_position = att_position
    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward



def beam_decode(self,target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = self.beam_size
    topk = beam_width
    decoded_batch = []
    attentions_batch = []
    att_pos_batch =[]
    dec_pred_output = []


    target_tensor = target_tensor.permute(1,0)
    B = target_tensor.size(0) # batch_size
    self.dec.all_pt = torch.empty((1, 0), device=self.device)
    logging.info("Beam searching")
    # Decoding goes sentence by sentence
    for idx in range(B):
        print(f"\r--- Sample {idx+1}/{B} ---", end="")
        x = self.x[:,idx,:].unsqueeze(1)
        enc_masks = torch.zeros((x.shape[0],1)) # (seq_len,batch_size)
        enc_masks[:self.src_lens[idx],0]=1
        self.attention_weights_list = []  # torch.empty((src_len,batch_size,trg_len))
        self.pt = torch.zeros((1,), device=self.device)
        self.dec.pt = self.pt
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))
        thr = random.random()
        teacher_force_ratio = 0
        # Start with the start of the special token <sos>
        dec_pred_output.append(torch.ones((1,1), dtype=torch.int,device=self.device))  # first tokens : <sos> index : 1
        decoder_input = target_tensor[idx].unsqueeze(0) if thr < teacher_force_ratio else dec_pred_output[idx]

        # starting node -  hidden vector, previous node, word id, logp, length

        node = BeamSearchNode(hiddenstate=decoder_hidden, previousNode=None, wordId=decoder_input.item(),
                              logProb=0,length=1,att_weights=None,att_position=None)

        nodes = PriorityQueue()
        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1
        # start beam search
        while True:
        #for idx in range(trg_len-1):
            # give up when decoding takes too long
            if qsize > 100: break
            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            #print(f"next word {idx}-->",n.wordid)
            if n.wordid == 2 and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            """"--------------------RUN MODEL PREDICT------------------------"""
            # decode for one step using decoder
            decoder_logits, decoder_hidden = self.dec(torch.tensor([[decoder_input]],device=self.device), decoder_hidden, encoder_output,enc_masks)

            att_weights = self.dec.weights_att.squeeze(2).squeeze(1)
            decoder_output = torch.log_softmax(decoder_logits,axis=-1)
            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output.squeeze(0), beam_width)
            nextnodes = []
            # beam loop
            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(hiddenstate=decoder_hidden, previousNode=n, wordId=decoded_t.item(),
                                      logProb=n.logp + log_p, length=n.leng + 1, att_weights=att_weights,
                                      att_position=int(self.dec.pt[0][0][0]) if 'local' in self.attention_type
                                                    else int(torch.max(att_weights)))

                score = -node.eval()
                nextnodes.append((score, node))
            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]
        utterances = []
        attentions = []
        positions = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            attention_per_beam = []
            positions_per_beam = []
            utterance.append(n.wordid)
            attention_per_beam.append(n.att_weights)
            positions_per_beam.append(n.att_position)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)
                if n.att_weights is not None :attention_per_beam.append(n.att_weights)
                if n.att_position is not None :positions_per_beam.append(n.att_position)

            assert n.att_weights is None
            # reverse words to have the correct order eos->sos >> sos->eos
            utterance = utterance[::-1]
            attention_per_beam = attention_per_beam[::-1]
            positions_per_beam = positions_per_beam[::-1]

            utterances.append(utterance)
            attentions.append(attention_per_beam)
            positions.append(positions_per_beam)

        decoded_batch.append(utterances)
        attentions_batch.append(attentions)
        att_pos_batch.append(positions)

    return decoded_batch,attentions_batch,att_pos_batch

if __name__=="__main__":
    hidden_size = 64
    embedding_dim = 64
    #decoder = seq2seq(642, hidden_size, embedding_dim, num_layers=1, device=device,
    #                  bidirectional=False, attention="local", mask=True, beam_size=2).to(device)
