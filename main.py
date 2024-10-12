# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : main.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: The program of calling all functions, such as fit,predict

from data.build_dataset import BuildDatabase
from models.fit import Fit
from models.predict import Test
from models.echo import echo

if __name__ == '__main__':
    # data precess
    # database = BuildDatabase(path_file='../Dataset/paths.log', dataset_path="dataset/Dataset", num_of_cores=32)
    # database.build()

    # model training
    # f = Fit(dataset_path='dataset/Dataset/all_graphs.bin', output_files='out_put/train')
    # f.fit()

    # # model test
    t = Test(path_file="./paths.log", output_files='out_put/att')
    t.output()

    # t = echo(path_file='../Dataset/paths.log', output_files='out_put/datasetpredict')
    # t.output()

