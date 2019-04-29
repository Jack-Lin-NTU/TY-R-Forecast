from argstools.argstools import args
from dataseters.trajGRU_all_data import TyDataset
from torchvision.transforms import Compose
args.load_all_data = True
a = TyDataset(args)
breakpoint()
print(a[0]['inputs'].shape)
