"""
- This __init__.py file is used to expose the classes defined in the architecture subpackage at the sub-level, 
- so that we can import them directly from the package without needing to specify the submodule. 
- This makes it more convenient for users of the package to access these classes without having to remember the specific submodule they are located in. 
- For example, instead of importing Mine_GradientDescent_LinaerRgression using
 from deep_learning_package.architecture.gradient_dsc_mine import Mine_GradientDescent_LinaerRgression
- we can simply import it using from deep_learning_package import Mine_GradientDescent_LinaerRgression after exposing it in the __init__.py file of the architecture subpackage.
- This approach helps to keep the package organized and user-friendly, allowing for easier access to the classes and functions defined in the submodules.
"""
from .gradient_dsc_mine import Mine_GradientDescent_LinaerRgression
from .stochasticGD import StocasticGD
from .logisticregressionGD import LogisticRegressionGD

# from .model_def import *

print(f"architecture package imported successfully: {StocasticGD} \n and Expose at Sub-level: Put this inside src/deep_learning_package/architecture/__init__.py")
