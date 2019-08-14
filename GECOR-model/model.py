import torch
import random
import numpy as np
from config import global_config as cfg
from reader import CamRest676Reader, get_glove_matrix
from GECOR_model import GECOR, cuda_
from torch.optim import Adam
from torch.autograd import Variable
from reader import pad_sequences
import argparse, time
from metric import CamRestEvaluator
import logging


class Model:
    def __init__(self, dataset):
        reader_dict = {
            'camrest': CamRest676Reader
        }
        model_dict = {
            'GECOR': GECOR
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                               hidden_size=cfg.hidden_size,
                               vocab_size=cfg.vocab_size,
                               layer_num=cfg.layer_num,
                               dropout_rate=cfg.dropout_rate,
                               z_length=cfg.z_length,
                               max_ts=cfg.max_ts,
                               beam_search=cfg.beam_search,
                               beam_size=cfg.beam_size,
                               eos_token_idx=self.reader.vocab.encode('EOS_M'),
                               vocab=self.reader.vocab,
                               teacher_force=cfg.teacher_force,
                               degree_size=cfg.degree_size)
        self.EV = evaluator_dict[dataset]  # evaluator class
        if cfg.cuda: self.m = self.m.cuda()
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']
        u_complete_input_py = py_batch['user_complete']
        u_complete_len_py = py_batch['u_complete_len']
        history_dialogue_py = py_batch['history_dialogue']
        history_dial_len_py = py_batch['history_dial_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2  # unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        u_complete_input_np = pad_sequences(u_complete_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        history_dialogue_np = pad_sequences(history_dialogue_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))

        u_complete_len = np.array(u_complete_len_py)
        history_dial_len = np.array(history_dial_len_py)
        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        u_complete_input = cuda_(Variable(torch.from_numpy(u_complete_input_np).long()))
        history_dialogue = cuda_(Variable(torch.from_numpy(history_dialogue_np).long()))
        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        z_input = cuda_(Variable(torch.from_numpy(z_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))

        kw_ret['z_input_np'] = z_input_np

        return history_dialogue_np, history_dialogue, history_dial_len, u_complete_input, u_complete_input_np, u_complete_len, u_input, u_input_np, z_input, \
               m_input, m_input_np, u_len, m_len, degree_input, kw_ret

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        prev_max_success = 0.1
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()

                    history_dialogue_np, history_dialogue, history_dial_len, u_complete_input, u_complete_input_np, \
                    u_complete_len, u_input, u_input_np, z_input, m_input, m_input_np, u_len, m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)

                    loss, turn_states = \
                        self.m(history_dialogue_np=history_dialogue_np, history_dialogue=history_dialogue,
                               history_dial_len=history_dial_len, u_complete_input=u_complete_input,
                               u_complete_input_np=u_complete_input_np, u_complete_len=u_complete_len,
                               u_input=u_input, u_input_np=u_input_np, turn_states=turn_states,
                               u_len=u_len, mode='train')
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 5.0)
                    optim.step()
                    sup_loss += loss.data.cpu().numpy()[0]
                    sup_cnt += 1
                    # logging.debug('loss:{} grad:{}'.format(loss.data[0], grad))
                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss, resolution_bleu_score, user_f1, user_accuracy = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time()-sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            ''' Accuracy 为主要评价指标 '''
            print('eval_res:', user_accuracy)
            print('prev_max:', prev_max_success)
            if user_accuracy >= prev_max_success:
                self.save_model(epoch)
                print('model has been saved.')
                prev_max_success = user_accuracy

            if valid_loss < prev_min_loss:
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))
        print('max:', prev_max_success)
        self.load_model(path=cfg.model_path)
        self.eval(data='test')

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                history_dialogue_np, history_dialogue, history_dial_len, u_complete_input, u_complete_input_np, u_complete_len, u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch, prev_z)
                c_idx, turn_states = \
                    self.m(history_dialogue_np=history_dialogue_np, history_dialogue=history_dialogue,
                           history_dial_len=history_dial_len, u_complete_input=u_complete_input,
                           u_complete_input_np=u_complete_input_np, u_complete_len=u_complete_len,
                           u_input=u_input, u_input_np=u_input_np, turn_states=turn_states,
                           u_len=u_len, mode='test')
                self.reader.wrap_result(turn_batch, c_idx)
        ev = self.EV(result_path=cfg.result_path)
        resolution_bleu_score, user_f1, user_accuracy = ev.run_metrics()
        self.m.train()
        return resolution_bleu_score, user_f1, user_accuracy

    def validate(self, data='test'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                history_dialogue_np, history_dialogue, history_dial_len, u_complete_input, u_complete_input_np, \
                u_complete_len, u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch)

                loss, turn_states = self.m(history_dialogue_np=history_dialogue_np,
                                                   history_dialogue=history_dialogue, history_dial_len=history_dial_len,
                                                   u_complete_input=u_complete_input,
                                                   u_complete_input_np=u_complete_input_np,
                                                   u_complete_len=u_complete_len, u_input=u_input,
                                                   turn_states=turn_states, u_input_np=u_input_np, u_len=u_len,
                                                   mode='train')
                sup_loss += loss.data[0]
                sup_cnt += 1
                logging.debug(
                    'loss:{}'.format(loss.data[0]))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        print('result preview...')
        resolution_bleu_score, user_f1, user_accuracy = self.eval(data='test')
        self.m.train()
        return sup_loss, unsup_loss, resolution_bleu_score, user_f1, user_accuracy

    def save_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path)
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.history_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.r_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])
        print('total trainable params: %d' % param_cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.init_handler(args.model)

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.debug(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.model.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        m.eval()


if __name__ == '__main__':
    main()
