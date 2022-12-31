

from _preprocess.process_functions import *


class Processor:

    def create_df(self, *args, **kwargs):
        return create_df(*args, **kwargs)

    def sliding_window(self, *args, **kwargs):
        return sliding_window(*args, **kwargs)

    def split_dataset_for_train_val_test(self, *args, **kwargs):
        return split_dataset_for_train_val_test(*args, **kwargs)

    def split_train_valid_test_by_tuple(self, *args, **kwargs):
        return split_train_valid_test_by_tuple(*args, **kwargs)

    def allocate_abnormal_data(self, *args, **kwargs):
        return allocate_abnormal_data(*args, **kwargs)

    def list_chain_custom(self, *args, **kwargs):
        return list_chain_custom(*args, **kwargs)

    def cal_stat_results(self, *args, **kwargs):
        return cal_stat_results(*args, **kwargs)

    def mse_loss(self, *args, **kwargs):
        return mse_loss(*args, **kwargs)