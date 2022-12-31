
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from _preprocess.process_class import Processor
from _model.doc2vec import DocToVec


class Orchestro(Processor,
                DocToVec):

    def __init__(self, window_params, etc_params, *args, **kwargs):
        super(Orchestro, self).__init__(*args, **kwargs)

        self.window_params = window_params
        self.etc_params = etc_params

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

        # make_embed_df
        embed_df = pd.DataFrame([doc_inferred_vec, label_list], columns=['EventId', 'Label'])

        normal_split_list = self.split_dataset_for_train_val_test(deeplog_df,
                                                                label_category=0,
                                                                split_n=self.etc_params['split_n'])
        normal_splited = self.split_train_valid_test_by_tuple(normal_split_list,
                                                              split_ratio=self.etc_params['split_data_ratio'])
        train_n, valid_n, test_n, rest_n = normal_splited[0], normal_splited[1], normal_splited[2], normal_splited[3]

        # split dataset for abnormal
        abnormal_split_list = self.split_dataset_for_train_val_test(deeplog_df,
                                                                  label_category=1,
                                                                  split_n=self.etc_params['split_n'])

        # allocate abnormal to normal
        abnormal_allocated = self.allocate_abnormal_data(abnormal_split_list, catergory_n=3)
        train_ab, valid_ab, test_ab = abnormal_allocated[0], abnormal_allocated[1], abnormal_allocated[2]

        # allocate rest to normal
        rest_allocated = self.allocate_abnormal_data(rest_n, catergory_n=3)
        train_rest, valid_rest, test_rest = rest_allocated[0], rest_allocated[1], rest_allocated[2]



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

    # run
    self = Orchestro(window_params=window_params,
                     etc_params=etc_params,
                     doc2vec_params=doc2vec_params)
