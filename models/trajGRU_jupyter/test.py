from tools.datasetGRU import *


if __name__ == '__main__':
    transform = Compose([ToTensor(), Normalize(max_values=args.max_values, min_values=args.min_values)])
    a = TyDataset(ty_list = args.ty_list,
                  input_frames = args.input_frames,
                  target_frames = args.target_frames,
                  train = True,
                  input_with_grid = args.input_with_grid,
                  input_with_QPE = args.input_with_QPE,
                  transform=transform
                 )
    
    print(a[0]['input'].shape)
