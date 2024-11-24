import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
from torch import Tensor
from torch.autograd import Variable
import glob
from os.path import join
import cv2
import pyiqa
from natsort import natsort
import argparse
import pandas as pd

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

if __name__ == '__main__':
    # test_images

    parser = argparse.ArgumentParser(description='NonPair Data Evaluation')
    parser.add_argument('--metric_name', type=str, default=['niqe']) 
    parser.add_argument('--model_name', type=str, default=['BiFormer-T', 'BiFormer-B']) 

    parser.add_argument('--dataset', type=str, default=['DICM', 'LIME', 'MEF', 'NPE', 'VV'])
    parser.add_argument('--base_result_path', type=str, default='./results_BiFormer/NonPair/')

    args = parser.parse_args()
    print("\nAvaliable Metrics:", pyiqa.list_models(), '\n----------------------------------------------\n')

    results_df = pd.DataFrame(columns=['Method', 'Dataset', 'Metric', 'Value'])

    for model_name in args.model_name:
        for dataset in args.dataset:
            result_path = os.path.join(args.base_result_path, model_name, dataset)
            result_path += '/'

            for metric_name in args.metric_name:
                iqa_metric = pyiqa.create_metric(metric_name).to('cuda')

                niqe_values = []
                paths_A = ''

                if dataset in ['DICM', 'LIME', 'MEF', 'NPE', 'VV']:
                    paths_A = fiFindByWildcard(os.path.join(result_path, f'*.png'))
                    if len(paths_A) == 0:
                        paths_A = fiFindByWildcard(os.path.join(result_path, f'*.jpg'))
                        if len(paths_A) == 0:
                            paths_A = fiFindByWildcard(os.path.join(result_path, f'*.bmp'))

                for idx, pathA in enumerate(paths_A):
                    img = cv2.imread(join(pathA))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    I1 = Variable(Tensor(np.array((img - img.min()) / (img.max()- img.min())))).unsqueeze(0).to('cuda')
                    I1 = I1.permute(0, 3, 1, 2)
                    niqe_value = iqa_metric(I1).detach().cpu().numpy()

                    niqe_values.append(niqe_value)

                avg_niqe_value = "{:.3f}".format(np.mean(np.array(niqe_values)))

                results_df = pd.concat([results_df, pd.DataFrame({'Method': model_name, 'Dataset': dataset, 'Metric': metric_name, 'Value': avg_niqe_value}, index=[0])])
        pivot_df = results_df.pivot_table(values='Value', index=['Method'], columns=['Dataset', 'Metric'])
    print(pivot_df)
    print("----------------------------------------------\n\n")
