from .baseline import BaselineModel
from .item_cf import ItemItemCF
from .matrix_factorization import MatrixFactorization

__all__ = ["BaselineModel", "ItemItemCF", "MatrixFactorization"]

MODEL_REGISTRY = {
    "baseline": BaselineModel,
    "itemcf": ItemItemCF,
    "mf": MatrixFactorization,
}
