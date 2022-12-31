
import pandas as pd
import numpy as np
import random


def create_df(df):
    df["datetime"] = pd.to_datetime(df["Date"] +" "+ df["Time"])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    df["Label"] = df["Label"]
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)
    return df


def sliding_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
#             print("process {} time window".format(num_session), end='\r')
            pass

    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index: end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index: end_index].values,
            max(label_data[start_index:end_index]),
            logkey_data[start_index: end_index].values,
            dt
        ])

    assert len(start_end_index_pair) == len(new_data)
#     print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return pd.DataFrame(new_data, columns=raw_data.columns)


def split_dataset_for_train_val_test(deeplog_df, label_category=0, split_n=10):
    deeplog_shuffled = deeplog_df[deeplog_df['Label'] == label_category].sample(frac=1)
    df_splits = np.array_split(deeplog_shuffled, split_n)

    return df_splits


def split_train_valid_test_by_tuple(data_list, split_ratio):

    datasets = []
    for i in range(len(split_ratio)):

        if i == 0:
            start = 0
            end = split_ratio[i]
        else:
            start = start + split_ratio[i-1]
            end = start + split_ratio[i]

        data = data_list[start:end]
        datasets.append(data)

    return datasets


def allocate_abnormal_data(data, catergory_n):

    data_indices = [i for i in range(len(data))]
    category_list = [x for x in range(catergory_n)]
    rest_n = len(data)
    data_selected_list = []
    for i in range(len(category_list)):
        select_n = random.sample([x for x in range(rest_n)], 1)[0]

        if i != len(category_list)-1:
            data_selected = random.sample(data_indices, select_n)
            if len(data_selected) > 0:
                for x in data_selected:
                    data_indices.remove(x)

        else:
            data_selected = data_indices
        rest_n = len(data_indices)

        data_selected_list.append(data_selected)

    df_selected_list = []
    for index in data_selected_list:
        dfs = [data[j] for j in index]
        df_selected_list.append(dfs)

    return df_selected_list

