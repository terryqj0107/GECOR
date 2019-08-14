import logging
import time


class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 0        
        self.eos_c_token = 'EOS_U'
        self.beam_len_bonus = 0.5
        self.mode = 'unknown'
        self.m = 'GECOR'
        self.prev_z_method = 'none'
        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'gecor-camrest': self._camrest_gecor_init
        }
        init_method[m]()

    def _camrest_gecor_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = (4, 0, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.entity = './data/CamRest676/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRestDB.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 1
        self.z_length = 8
        self.degree_size = 5
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100  # triggered by early stop
        self.rl_epoch_num = 20
        self.cuda = True
        self.spv_proportion = 100
        self.max_ts = 40
        self.early_stop_count = 5
        self.new_vocab = True
        self.data = './data/CamRest676/CamRest676_annotated.json'
        self.vocab_path = './vocab/CamRest676_annotated_complete.pkl'

        self.model_path = './models/camrest_annotated_mixed+greedy+gated_copy_history+modifiedloss.pkl'
        # self.model_path = './models/camrest_annotated_ellipsis_recovery+greedy+gated_copy_history+modifiedloss.pkl'
        # self.model_path = './models/camrest_annotated_coreference_resolution+greedy+gated_copy_history+modifiedloss.pkl'

        self.result_path = './results/camrest_annotated_mixed+greedy+gated_copy_history+modifiedloss.csv'
        # self.result_path = './results/camrest_annotated_ellipsis_recovery+greedy+gated_copy_history+modifiedloss.csv'
        # self.result_path = './results/camrest_annotated_coreference_resolution+greedy+gated_copy_history+modifiedloss.csv'

        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)


global_config = _Config()

