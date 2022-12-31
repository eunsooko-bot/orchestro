
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from itertools import chain
import json

from _preprocess.process_class import Processor
from _model.doc2vec import DocToVec
from _model.VAE import Tf_VAE


class Orchestro(Processor,
                DocToVec):

    def __init__(self, window_params, etc_params, tf_params, *args, **kwargs):
        super(Orchestro, self).__init__(*args, **kwargs)

        # params
        self.window_params = window_params
        self.etc_params = etc_params
        self.tf_params = tf_params

        # model
        self.VAE = Tf_VAE(tf_params)

    def run_vae_process(self):

        # dataset
        self.train_emb, self.valid_emb, self.test_emb, \
        self.train_label, self.valid_label, self.test_label, \
        self.info_dic = self.prepare_dataset()

        # model fit
        self.VAE.build_network(original_dim=len(self.train_emb[0]))
        self.VAE.model_fit(self.train_emb)

        # validation prediction
        self.valid_pred = self.VAE.model_predict(self.valid_emb)
        valid_error = self.mse_loss(self.valid_emb, self.valid_pred)

        # test prediction
        self.test_pred = self.VAE.model_predict(self.test_emb)
        test_error = self.mse_loss(self.test_emb, self.test_pred)

        # get stats dic
        self.stats_dic = self.cal_stat_results(valid_error, test_error, self.test_label)

        return self.info_dic, self.stats_dic

    def inject_data(self, path):

        log_structured = pd.read_csv(path)
        log_structured_sorted = log_structured.sort_values(by=["Date", "Time"])
        log_structured_sorted['Label'] = [1 if i == -1 else 0 for i in log_structured_sorted['Label']]
        log_df = self.create_df(log_structured_sorted)

        deeplog_df = self.sliding_window(log_df[["timestamp", "Label", "EventId", "deltaT"]],
                                        para=self.window_params)

        return deeplog_df

    def prepare_dataset(self):

        # inject dataset
        deeplog_df = self.inject_data(self.etc_params['data_path'])
        event_id_list = deeplog_df['EventId'].to_list()
        label_list = deeplog_df['Label'].to_list()

        # load doc2vec model
        doc2vec_model = self.load_model_artifact(save_path=self.doc2vec_params['save_path'])

        # transform data using model
        doc_inferred_vec = np.asarray([doc2vec_model.infer_vector(document) for document in event_id_list])
        self.doc_inferred_vec = doc_inferred_vec

        # split dataset
        normal_split_list = self.split_dataset_for_train_val_test(doc_inferred_vec, label_list,
                                                                  label_category=0,
                                                                  split_n=self.etc_params['split_n'])
        normal_splited = self.split_train_valid_test_by_tuple(normal_split_list,
                                                              split_ratio=self.etc_params['split_data_ratio'])
        train_n, valid_n, test_n, rest_n = normal_splited[0], normal_splited[1], normal_splited[2], normal_splited[3]

        # split dataset for abnormal
        abnormal_split_list = self.split_dataset_for_train_val_test(doc_inferred_vec, label_list,
                                                                    label_category=1,
                                                                    split_n=self.etc_params['split_n'])

        # allocate abnormal to normal
        abnormal_allocated = self.allocate_abnormal_data(abnormal_split_list, catergory_n=3)
        train_ab, valid_ab, test_ab = abnormal_allocated[0], abnormal_allocated[1], abnormal_allocated[2]

        # allocate rest to normal
        rest_allocated = self.allocate_abnormal_data(rest_n, catergory_n=3)
        train_rest, valid_rest, test_rest = rest_allocated[0], rest_allocated[1], rest_allocated[2]

        # assembly
        tr_a = self.list_chain_custom(train_n)
        tr_b = self.list_chain_custom(train_ab)
        tr_c = self.list_chain_custom(train_rest)
        va_a = self.list_chain_custom(valid_n)
        va_b =  self.list_chain_custom(valid_ab)
        va_c =   self.list_chain_custom(valid_rest)
        te_a =  self.list_chain_custom(test_n)
        te_b = self.list_chain_custom(test_ab)
        te_c = self.list_chain_custom(test_rest)

        train_emb = tr_a + tr_b + tr_c
        valid_emb = va_a + va_b + va_c
        test_emb = te_a + te_b + te_c

        # label
        train_label= [1 for _ in range(len(tr_a))] + [0 for i in range(len(tr_b))] + [1 for i in range(len(tr_c))]
        valid_label = [1 for _ in range(len(va_a))] + [0 for i in range(len(va_b))] + [1 for i in range(len(va_c))]
        test_label = [1 for _ in range(len(te_a))] + [0 for i in range(len(te_b))] + [1 for i in
                                                                                           range(len(te_c))]

        # data portion info
        info_dic = {
            "train": {'normal': len(train_n)+len(train_rest),
                      'abnormal': len(train_ab)},
            "valid": {'normal': len(valid_n)+len(valid_rest),
                      'abnormal': len(valid_ab)},
            "test": {'normal': len(test_n) + len(test_rest),
                      'abnormal': len(test_ab)},
        }

        return np.array(train_emb), np.array(valid_emb), np.array(test_emb), train_label, valid_label, test_label, info_dic


if __name__ == "__main__":

    # params
    window_params = {
        "window_size": 5,
        "step_size": 1
    }

    doc2vec_params = {
        'max_epochs': 20,
        'save_path': 'artifact/Doc2VecModel_templates_221116'
    }

    etc_params = {
        "data_path": 'data/openstack_labeled2.log_structured.csv',
        "split_n": 10,
        "split_data_ratio": (3,3,2,2)
    }

    tf_params = {
        "epochs": 10,
        "learning_rate": 0.0001,
        "shuffle": True,
        "batch_size": 64

    }

    # run
    self = Orchestro(window_params=window_params,
                     etc_params=etc_params,
                     doc2vec_params=doc2vec_params,
                     tf_params=tf_params)
