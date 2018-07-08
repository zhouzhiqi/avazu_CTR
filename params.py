
class flags(object):
    def __init__(self, file_name, onehot_cat):
        self.file_name = file_name
        self.onehot_cat = onehot_cat
        self.data_dir = '../data/project_2/data/{0}/'.format(self.onehot_cat)
        self.output_dir = '../data/project_2/output/{0}/'.format(self.onehot_cat)
        self.model_dir = '../data/project_2/models/{0}/'.format(self.onehot_cat)

class params(object):
    def __init__(self, onehot_name):
        self.threshold = 10
        self.chunksize = 1e3
        self.num_trees = 50
        self.deep = 9
        self.split = '='
        # ['A_cat', 'A_hour', 'A_xgb', 
        #  'B_cat', 'C_his']
        self.onehot_name = onehot_name
        self.lr_C = 1
