import torch
import torchtext


def decode_numeric_sentence(self, list_int, remove_sos_eos=False, ignore_pad=False):
    " Return human read-able sentence"
    if remove_sos_eos:
        sos_id = 1 if self.token_to_idx[
                          "<sos>"] in list_int else 0  # exclude sos from the targets <-- some predicted sentences does not include sos or eos
        try:
            eos_id = list_int.index(self.token_to_idx["<eos>"])
            return " ".join([self.idx_to_token[idx] for idx in list_int[sos_id:eos_id + 1]
                             if idx != self.token_to_idx["<pad>"] or not ignore_pad])
        except ValueError:
            # WHEN THE EOS TOKEN IS NOT GENERATE IN THE PREDICTION
            return " ".join([self.idx_to_token[idx] for idx in list_int[sos_id:]
                             if idx != self.token_to_idx["<pad>"] or not ignore_pad])

    else:
        try:
            pad_id = list_int.index(self.token_to_idx["<pad>"])
            return " ".join([self.idx_to_token[idx] for idx in list_int[:pad_id] if
                             idx != self.token_to_idx["<pad>"] or not ignore_pad])
        except ValueError:
            return " ".join([self.idx_to_token[idx] for idx in list_int
                             if idx != self.token_to_idx["<pad>"] or not ignore_pad])

def language_tokenizer(self, sentence):
    sentence = str(sentence)  # numpy str --> str
    return [str(token) for token in self.spacy_eng.tokenizer(sentence)]


def decode_predictions_and_compute_bleu_score(output,trg,vocab_obj,num_grams=4,batch_first=False,multiple_references = False):
        # Decode Predictions
        scores = torch.nn.Softmax(dim=-1)(output)
        numeric_predictions = scores.argmax(dim=-1)
        if batch_first :
            numeric_predictions = numeric_predictions.permute(1,0)
            trg = trg.permute(1,0) if not multiple_references else trg
        predicted_sentences,output_prediction  = [],[]
        _dec_numeric_sentence = vocab_obj.decode_numeric_sentence
        for k in range(numeric_predictions.size(1)):
            predicted_sentences += [_dec_numeric_sentence(numeric_predictions[:,k].cpu().numpy().tolist(),remove_sos_eos=True).split(" ") ]

        if multiple_references:
            input_sentences = [[_dec_numeric_sentence(ref,remove_sos_eos=True).split(" ") for ref in refs] for refs in trg ]
        else:
            input_sentences = [[_dec_numeric_sentence(inp_.tolist(),remove_sos_eos=True).split(" ")] for inp_ in trg.T.cpu().numpy() ]

        a = 1/num_grams
        bleu_score = torchtext.data.metrics.bleu_score(candidate_corpus = predicted_sentences,
                                          references_corpus = input_sentences,
                                          max_n=num_grams, weights=[a]*num_grams)

        return bleu_score,predicted_sentences,input_sentences

def decode_reference(vocab_obj,trg,multiple_references = False):
        _dec_numeric_sentence = vocab_obj.decode_numeric_sentence

        if multiple_references:
            input_sentences = [[_dec_numeric_sentence(ref,remove_sos_eos=True).split(" ") for ref in refs] for refs in trg ]
        else:
            input_sentences = [[_dec_numeric_sentence(inp_.tolist(),remove_sos_eos=True).split(" ")] for inp_ in trg.T.cpu().numpy() ]

        return input_sentences