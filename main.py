
from tqdm import tqdm
import os
import json
from _project.orchestro import Orchestro


if __name__=="__main__":

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
        "split_data_ratio": (3, 3, 2, 2)
    }

    tf_params = {
        "epochs": 100,
        "learning_rate": 0.0001,
        "shuffle": True,
        "batch_size": 64

    }

    #
    json_save_path = 'output'
    if not os.path.isdir(json_save_path):
        os.mkdir(json_save_path)

    # run
    self = Orchestro(window_params=window_params,
                     etc_params=etc_params,
                     doc2vec_params=doc2vec_params,
                     tf_params=tf_params)

    run_ = 100

    for i in tqdm(range(run_), desc='main.py'):

        info_dic, stats_dic = self.run_vae_process()

        # output
        with open(f"{json_save_path}/info_dic_{i}.json", 'w') as f:
            json.dump(self.info_dic, f)

        with open(f"{json_save_path}/stats_dic_{i}.json", 'w') as f:
            json.dump(self.stats_dic, f)

        print(f"Done for {i}")