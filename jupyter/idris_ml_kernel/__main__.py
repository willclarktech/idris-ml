"""Entry point: python -m idris_ml_kernel"""

from ipykernel.kernelapp import IPKernelApp
from .kernel import Idris2Kernel

IPKernelApp.launch_instance(kernel_class=Idris2Kernel)
