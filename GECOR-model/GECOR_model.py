import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from config import global_config as cfg
import copy, random, time, logging
from reader import pad_sequences


def cuda_(var):
    return var.cuda() if cfg.cuda else var


def toss_(p):
    return random.randint(0, 99) <= p


def nan(v):
    if type(v) is float:
        return v == float('nan')
    return np.isnan(np.sum(v.data.cpu().numpy()))


def get_sparse_input_aug(x_input_np):
    """
    sparse input of
    :param x_input_np: [T,B]
    :return: Numpy array: [B,T,aug_V]
    """
    ignore_index = [0]
    unk = 2
    result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                      dtype=np.float32)
    result.fill(1e-10)
    for t in range(x_input_np.shape[0]):
        for b in range(x_input_np.shape[1]):
            w = x_input_np[t][b]
            if w not in ignore_index:
                if w != unk:
                    result[t][b][x_input_np[t][b]] = 1.0
                else:
                    result[t][b][cfg.vocab_size + t] = 1.0
    result_np = result.transpose((1, 0, 2))
    result = torch.from_numpy(result_np).float()
    return result


def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal(hh[i:i+gru.hidden_size],gain=1)


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, normalize=True):
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B,T,H]
        attn_energies = self.score(hidden, encoder_outputs)
        normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context.transpose(0, 1), normalized_energy.squeeze(1)    # [1,B,H] , [B,T]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        energy = F.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        return energy


class SimpleDynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        init_gru(self.gru)
            
    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class HistoryEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        init_gru(self.gru)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = input_lens[sort_idx]
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs, hidden, embedded


class ResolutionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate, emb, vocab):
        super().__init__()
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, dropout=dropout_rate)
        self.proj = nn.Linear(hidden_size * 2, vocab_size)
        self.emb = emb
        self.attn_u = Attn(hidden_size)
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate
        init_gru(self.gru)
        self.inp_dropout = nn.Dropout(self.dropout_rate)
        self.vocab = vocab
        self.proj_pgen1 = nn.Linear(hidden_size * 2 + embed_size, 1)

    def get_sparse_selective_input(self, x_input_np):
        result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]),
                          dtype=np.float32)
        result.fill(1e-10)
        reqs = ['address', 'phone', 'postcode', 'pricerange', 'area']
        for t in range(x_input_np.shape[0] - 1):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if word in reqs:
                    slot = self.vocab.encode(word + '_SLOT')
                    result[t + 1][b][slot] = 1.0
                else:
                    if w == 2 or w >= cfg.vocab_size:
                        result[t + 1][b][cfg.vocab_size + t] = 5.0
                    else:
                        result[t + 1][b][w] = 1.0
        result_np = result.transpose((1, 0, 2))  # [B, T, V+T]
        result = torch.from_numpy(result_np).float()
        return result

    def forward(self, history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_t_input, last_hidden):
        c_embed = self.emb(c_t_input)  # [1,B,E]
        u_context, att = self.attn_u(last_hidden, u_enc_out)  # u_context: [1,B,H]   att: [B,Tu]   u_enc_out : [Tu,B,H]
        gru_in = torch.cat([c_embed, u_context], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([u_context, gru_out], 2)).squeeze(0)  # [B,V]

        sparse_u_input = Variable(self.get_sparse_selective_input(history_dialogue_np), requires_grad=False)  # [B,Tu,V+Tu]
        u_copy_score = F.tanh(self.proj_copy2(history_enc_out.transpose(0, 1)))
        u_copy_score = torch.matmul(u_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        u_copy_score = u_copy_score.cpu()
        u_copy_score_max = torch.max(u_copy_score, dim=1, keepdim=True)[0]
        u_copy_score = torch.exp(u_copy_score - u_copy_score_max)  # [B,Tu]
        u_copy_score = torch.log(torch.bmm(u_copy_score.unsqueeze(1), sparse_u_input)).squeeze(
            1) + u_copy_score_max
        u_copy_score = cuda_(u_copy_score)  # [B,V+Tu]

        prob_gen_1 = F.sigmoid(self.proj_pgen1(torch.cat([u_context, last_hidden, c_embed], 2)).squeeze(0))  # [B,1]
        prob_gen_2 = 1 - prob_gen_1
        scores = F.softmax(torch.cat([gen_score * prob_gen_1, u_copy_score * prob_gen_2], dim=1), dim=1)  # [B, V + V+Tu]

        # scores = F.softmax(torch.cat([gen_score, u_copy_score], dim=1), dim=1)  # [B, V + V+Tu]
        gen_score, u_copy_score = scores[:, :cfg.vocab_size], scores[:, cfg.vocab_size:]    # [B,V]  , [B,V+Tu]
        proba = gen_score + u_copy_score[:, :cfg.vocab_size]  # [B,V] 词表范围内的词
        proba = torch.cat([proba, u_copy_score[:, cfg.vocab_size:]], 1)  # 拼接来自copy范围的概率分布
        return proba, last_hidden, gru_out


class GECOR(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, layer_num, dropout_rate, z_length,
                 max_ts, beam_search=False, teacher_force=100, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.emb = nn.Embedding(vocab_size, embed_size)

        self.history_encoder = HistoryEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.u_encoder = SimpleDynamicEncoder(vocab_size, embed_size, hidden_size, layer_num, dropout_rate)
        self.r_decoder = ResolutionDecoder(embed_size, hidden_size, vocab_size, degree_size, dropout_rate,
                                           self.emb, self.vocab)  # 用于完成指代消解和省略补全任务
        self.embed_size = embed_size
        self.max_ts = max_ts
        self.beam_search = beam_search
        self.teacher_force = teacher_force
        self.c_dec_loss = nn.NLLLoss(ignore_index=0)
        self.saved_log_policy = []

        if self.beam_search:
            self.beam_size = kwargs['beam_size']
            self.eos_token_idx = kwargs['eos_token_idx']

    def forward(self, history_dialogue_np, history_dialogue, history_dial_len, u_complete_input, u_complete_input_np,
                u_complete_len, u_input, u_input_np, u_len, turn_states, mode):
        if mode == 'train' or mode == 'valid':
            pc_dec_proba, turn_states = \
                self.forward_turn(history_dialogue_np=history_dialogue_np, history_dialogue=history_dialogue,
                                  history_dial_len=history_dial_len, u_complete_input=u_complete_input,
                                  u_complete_input_np=u_complete_input_np, u_complete_len=u_complete_len,
                                  u_input=u_input, u_len=u_len, mode='train', turn_states=turn_states,
                                  u_input_np=u_input_np)
            loss = self.supervised_loss(torch.log(pc_dec_proba), u_complete_input)
            return loss, turn_states

        elif mode == 'test':
            c_output_index, turn_states = \
                self.forward_turn(history_dialogue_np=history_dialogue_np, history_dialogue=history_dialogue,
                                  history_dial_len=history_dial_len, u_complete_input=u_complete_input,
                                  u_complete_input_np=u_complete_input_np, u_complete_len=u_complete_len,
                                  u_input=u_input, u_len=u_len, mode='test', turn_states=turn_states,
                                  u_input_np=u_input_np)
            return c_output_index, turn_states

    def forward_turn(self, history_dialogue_np, history_dialogue, history_dial_len, u_complete_input,
                     u_complete_input_np, u_complete_len, u_input, u_len, turn_states, mode, u_input_np):
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        :param u_input: [T,B]
        """
        batch_size = u_input.size(1)
        u_enc_out, u_enc_hidden, u_emb = self.u_encoder(u_input, u_len)
        history_enc_out, history_enc_hidden, history_emb = self.history_encoder(history_dialogue, history_dial_len)
        last_hidden = u_enc_hidden[:-1]
        c_tm1 = cuda_(Variable(torch.ones(1, batch_size).long()))  # GO token

        if mode == 'train':

            ''' 指代消解 & 省略补全   预测完整的用户语句 '''
            pc_dec_proba, c_dec_outs = [], []
            c_length = u_complete_input.size(0)  # Tc
            for t in range(c_length):
                teacher_forcing = toss_(self.teacher_force)
                proba, last_hidden, dec_out = self.r_decoder(history_dialogue_np, history_enc_out, u_enc_out,
                                                             u_input_np, c_tm1, last_hidden)
                if teacher_forcing:
                    c_tm1 = u_complete_input[t].view(1, -1)  # [1,B]
                else:
                    _, c_tm1 = torch.topk(proba, 1)
                    c_tm1 = c_tm1.view(1, -1)
                pc_dec_proba.append(proba)
                c_dec_outs.append(dec_out)
            c_dec_outs = torch.cat(c_dec_outs, dim=0)  # [Tc,B,H]
            pc_dec_proba = torch.stack(pc_dec_proba, dim=0)  # [Tc,B,V+Tc]

            return pc_dec_proba, None
        else:
            if mode == 'test':
                if not self.beam_search:
                    c_dec_outs, pc_dec_proba, last_hidden, c_output_index = \
                        self.c_greedy_decode(history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_tm1, last_hidden)
                else:
                    c_output_index = \
                        self.c_beam_search_decode(history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_tm1, last_hidden)

                return c_output_index, None

    def c_greedy_decode(self, history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_tm1, last_hidden):
        pc_dec_proba, c_dec_outs = [], []
        decoded = []
        for t in range(self.max_ts):
            proba, last_hidden, dec_out = \
                self.r_decoder(history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_tm1, last_hidden)
            ct_proba, ct_index = torch.topk(proba, 1)
            ct_index = ct_index.data.view(-1)
            decoded.append(ct_index.clone())
            for i in range(ct_index.size(0)):
                if ct_index[i] >= cfg.vocab_size:
                    ct_index[i] = 2  # unk
            c_tm1 = cuda_(Variable(ct_index).view(1, -1))
            pc_dec_proba.append(proba)
            c_dec_outs.append(dec_out)
        c_dec_outs = torch.cat(c_dec_outs, dim=0)  # [max_Ts,B,H]
        pc_dec_proba = torch.stack(pc_dec_proba, dim=0)  # [max_Ts,B,V]
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        c_output_index = [list(_) for _ in decoded]  # [B, max_Ts]
        return c_dec_outs, pc_dec_proba, last_hidden, c_output_index

    def c_beam_search_decode_single(self, history_dialogue_np, history_enc_out_s, u_enc_out, c_tm1, u_input_np, last_hidden):
        batch_size = history_enc_out_s.size(1)
        if batch_size != 1:
            raise ValueError('"Beam search single" requires batch size to be 1')

        class BeamState:
            def __init__(self, score, last_hidden, decoded, length):
                """
                Beam state in beam decoding
                :param score: sum of log-probabilities
                :param last_hidden: last hidden
                :param decoded: list of *Variable[1*1]* of all decoded words
                :param length: current decoded sentence length
                """
                self.score = score
                self.last_hidden = last_hidden
                self.decoded = decoded
                self.length = length

            def update_clone(self, score_incre, last_hidden, decoded_t):
                decoded = copy.copy(self.decoded)
                decoded.append(decoded_t)
                clone = BeamState(self.score + score_incre, last_hidden, decoded, self.length + 1)
                return clone

        def beam_result_valid(decoded_t):

            return True

        def score_bonus(state, decoded):
            bonus = cfg.beam_len_bonus
            return bonus

        def soft_score_incre(score, turn):
            return score

        finished, failed = [], []
        states = []  # sorted by score decreasingly
        dead_k = 0
        states.append(BeamState(0, last_hidden, [c_tm1], 0))
        for t in range(self.max_ts):
            new_states = []
            k = 0
            while k < len(states) and k < self.beam_size - dead_k:
                state = states[k]
                last_hidden, c_tm1 = state.last_hidden, state.decoded[-1]
                proba, last_hidden, _ = \
                    self.r_decoder(history_dialogue_np, history_enc_out_s, u_enc_out, u_input_np, c_tm1, last_hidden)
                proba = torch.log(proba)
                ct_proba, ct_index = torch.topk(proba, self.beam_size - dead_k)  # [1,K]
                for new_k in range(self.beam_size - dead_k):
                    score_incre = soft_score_incre(ct_proba[0][new_k].data[0], t) + \
                                  score_bonus(state, ct_index[0][new_k].data[0])
                    if len(new_states) >= self.beam_size - dead_k and state.score + score_incre < new_states[-1].score:
                        break
                    decoded_t = ct_index[0][new_k]
                    if decoded_t.data[0] >= cfg.vocab_size:
                        decoded_t.data[0] = 2  # unk
                    if self.vocab.decode(decoded_t.data[0]) == cfg.eos_c_token:
                        if beam_result_valid(state.decoded):
                            finished.append(state)
                            dead_k += 1
                        else:
                            failed.append(state)
                    else:
                        decoded_t = decoded_t.view(1, -1)
                        new_state = state.update_clone(score_incre, last_hidden, decoded_t)
                        new_states.append(new_state)

                k += 1
            if self.beam_size - dead_k < 0:
                break
            new_states = new_states[:self.beam_size - dead_k]
            new_states.sort(key=lambda x: -x.score)
            states = new_states

            if t == self.max_ts - 1 and not finished:
                finished = failed
                # print('FAIL')
                if not finished:
                    finished.append(states[0])

        finished.sort(key=lambda x: -x.score)
        decoded_t = finished[0].decoded
        decoded_t = [_.view(-1).data[0] for _ in decoded_t]
        decoded_sentence = self.vocab.sentence_decode(decoded_t, cfg.eos_c_token)
        # print(decoded_sentence)
        generated = torch.cat(finished[0].decoded, dim=1).data  # [B=1, T]
        generated = generated[:, 1:]
        return generated

    def c_beam_search_decode(self, history_dialogue_np, history_enc_out, u_enc_out, u_input_np, c_tm1, last_hidden):
        vars = torch.split(history_enc_out, 1, dim=1), torch.split(u_enc_out, 1, dim=1),\
               torch.split(c_tm1, 1, dim=1), torch.split(last_hidden, 1, dim=1)
        decoded = []
        for i, (history_enc_out_s, u_enc_out_s, c_tm1_s, last_hidden_s) in enumerate(zip(*vars)):
            decoded_s = self.c_beam_search_decode_single(history_dialogue_np[:, i].reshape((-1, 1)),
                                                         history_enc_out_s, u_enc_out_s, c_tm1_s,
                                                         u_input_np[:, i].reshape((-1, 1)), last_hidden_s)
            decoded.append(decoded_s)
        return [list(_.view(-1)) for _ in decoded]

    def supervised_loss(self, pc_dec_proba, u_complete_input):
        # pc_dec_proba = pc_dec_proba[:, :, :cfg.vocab_size].contiguous()
        pc_dec_proba = pc_dec_proba.contiguous()
        c_loss = self.c_dec_loss(pc_dec_proba.view(-1, pc_dec_proba.size(2)), u_complete_input.view(-1))
        loss = c_loss
        return loss

    def self_adjust(self, epoch):
        pass
