from torch.nn.functional import relu, softmax, cross_entropy
from torch.optim import Adam
import os
from betweenness_centrality import BetweennessCentralityCalculator
from bfs_moments import BfsMomentsCalculator
from bilinear_model import LayeredBilinearModule
from dataset.dataset import BilinearDataset
from dataset.dataset_external_data import ExternalData
from feature_calculators import FeatureMeta
from multi_class_bilinear_activator import BilinearMultiClassActivator
from params.parameters import BilinearDatasetParams, BilinearActivatorParams, BilinearLayerParams, LinearLayerParams, \
    LayeredBilinearModuleParams, DEG, CENTRALITY, BFS, NORM_REDUCED, ExternalDataParams


# ---------------------------------------------------  PROTEIN ---------------------------------------------------------
class CoilDelAllExternalDataParams(ExternalDataParams):
    def __init__(self):
        super().__init__()
        self.GRAPH_COL = "g_id"
        self.NODE_COL = "node"
        self.FILE_NAME = "COIL-DEL_external_data_all.csv"
        self.EMBED_COLS = []
        self.VALUE_COLS = ["x", "y"]


class CoilDelDatasetAllParams(BilinearDatasetParams):
    def __init__(self):
        super().__init__()
        self.DATASET_NAME = "coil_del_train"
        self.DATASET_FILENAME = "COIL-DEL_all.csv"
        self.SRC_COL = "src"
        self.DST_COL = "dst"
        self.GRAPH_NAME_COL = "g_id"
        self.LABEL_COL = "label"
        self.PERCENTAGE = 1
        self.DIRECTED = False
        self.FEATURES = [DEG, CENTRALITY, BFS]


# ----------------------------------------------------------------------------------------------------------------------


class CoilDelBilinearLayerParams(BilinearLayerParams):
    def __init__(self, in_col_dim, ftr_len):
        super().__init__(in_col_dim, ftr_len)
        self.LEFT_LINEAR_ROW_DIM = in_col_dim   # should be equal to RIGHT_LINEAR_IN and FirstLayerModelParams::OUT_DIM
        self.LEFT_LINEAR_COL_DIM = 1            # out rows
        self.RIGHT_LINEAR_ROW_DIM = ftr_len     # should be equal to FirstLayerModelParams::ROW_DIM
        self.RIGHT_LINEAR_COL_DIM = 100         # out cols
        self.ACTIVATION_FUNC = softmax
        self.ACTIVATION_FUNC_ARGS = {"dim": 1}


class CoilDelLinearLayerParams(LinearLayerParams):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__(in_dim, out_dim, dropout)
        self.ROW_DIM = in_dim
        self.COL_DIM = out_dim
        self.ACTIVATION_FUNC = relu
        self.DROPOUT = dropout


class CoilDelLayeredBilinearModuleParams(LayeredBilinearModuleParams):
    def __init__(self, ftr_len=6, layer_dim=None, embed_vocab_dim=None):
        super().__init__(ftr_len, layer_dim, embed_vocab_dim)
        self.EMBED_DIMS = []
        self.NORM = NORM_REDUCED
        self.DROPOUT = 0
        self.LR = 1e-3
        self.OPTIMIZER = Adam
        self.WEIGHT_DECAY = 1e-2

        if layer_dim is None:
            self.NUM_LAYERS = 2
            self.LINEAR_PARAMS_LIST = [
                CoilDelLinearLayerParams(in_dim=ftr_len, out_dim=50, dropout=self.DROPOUT),
                CoilDelLinearLayerParams(in_dim=50, out_dim=25, dropout=self.DROPOUT),
                CoilDelLinearLayerParams(in_dim=50, out_dim=10, dropout=self.DROPOUT),
                CoilDelLinearLayerParams(in_dim=200, out_dim=1, dropout=self.DROPOUT)
            ]
        self.BILINEAR_PARAMS = CoilDelBilinearLayerParams(self.LINEAR_PARAMS_LIST[self.NUM_LAYERS - 1].COL_DIM,
                                                          self.LINEAR_PARAMS_LIST[0].ROW_DIM)


class CoilDelBilinearActivatorParams(BilinearActivatorParams):
    def __init__(self):
        super().__init__()
        self.DEV_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        self.LOSS = cross_entropy  # f.factor_loss  #
        self.BATCH_SIZE = 16
        self.EPOCHS = 500
        self.DATASET = "COIL_DEL - MultiClass"


if __name__ == '__main__':
    ext_train = ExternalData(CoilDelAllExternalDataParams())
    aids_train_ds = BilinearDataset(CoilDelDatasetAllParams(), external_data=ext_train)

    activator = BilinearMultiClassActivator(LayeredBilinearModule(CoilDelLayeredBilinearModuleParams(
        ftr_len=aids_train_ds.len_features, embed_vocab_dim=ext_train.len_embed())),
        CoilDelBilinearActivatorParams(), aids_train_ds)
    activator.train()
