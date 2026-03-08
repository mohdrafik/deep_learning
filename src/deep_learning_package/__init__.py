# Use the dot (.) to indicate "from this same folder"

from .general_utils import DataReader, UniversalPlotter
from .architecture import Mine_GradientDescent_LinaerRgression, StocasticGD, LogisticRegressionGD
# from .architecture model_def
from .model_architecture import DeepModel  
from .trainer import Trainer

# from .general_utils.data_read import DataReader
# from .general_utils.univershal_plotter import UniversalPlotter


__version__ = "0.1.0"