
import numpy as np
import gensim.models.doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class DocToVec:

    def __init__(self, doc2vec_params):
        self.doc2vec_params = doc2vec_params

    def __call__(self, event_id_list):
        Embed_documents = self.gen_embed_document(event_id_list)
        self.get_train_model(Embed_documents, self.doc2vec_params['max_epochs'])
        self.save_model_artifact(save_path=self.doc2vec_params['save_path'])

    def gen_embed_document(self, event_id_list):

        Embed_tuple = [tuple(data) for data in event_id_list]
        Uniqu_tuple = list(set(Embed_tuple))

        Tags_dict = {str(list(tuple_data)): "TypeID_{0}".format(index) for index, tuple_data in enumerate(Uniqu_tuple)}
        Tags_list = [Tags_dict[str(list(trains))] for trains in event_id_list]

        Embed_documents = [TaggedDocument(train_data, tags=window_tag) for window_tag, train_data in
                           zip(Tags_list, event_id_list)]

        return Embed_documents

    def get_train_model(self, Embed_documents, max_epochs):

        train_model = Doc2Vec(window=10, vector_size=300, alpha=0.025, min_alpha=0.025, min_count=2, dm=1, negative=5,
                              seed=9999)
        train_model.build_vocab(Embed_documents)

        for epoch in range(max_epochs):
            print("iteration {0}".format(epoch))
            train_model.train(Embed_documents, total_examples=train_model.corpus_count,
                              epochs=train_model.epochs)

            train_model.alpha -= 0.002
            train_model.min_alpha = train_model.alpha

        self.train_model = train_model

    def save_model_artifact(self, save_path):
        self.train_model.save(save_path)

    def load_model_artifact(self, save_path):
        train_model = gensim.models.Word2Vec.load(save_path)

        return train_model