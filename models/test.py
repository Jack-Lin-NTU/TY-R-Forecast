from src.utils.parser import get_args
from src.runs.mymodel import get_model, get_dataloader

args = get_args()
trainloader,_ = get_dataloader(args)
breakpoint()
model = get_model(args=args)