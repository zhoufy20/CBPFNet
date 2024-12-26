# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : main.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: The program of calling all functions, such as fit,predict

from data.build_dataset import BuildDatabase
from models.fit import Fit
from models.predict import Test
from models.echo import echo
from models.echoscore import Score

if __name__ == '__main__':
    # train data process
    # database = BuildDatabase(path_file='../Dataset/paths.log', dataset_path="dataset/traindataset", num_of_cores=32)
    # database.build()

    # model training
    f = Fit(dataset_path='dataset/traindataset/all_graphs.bin', output_files='out_put/train')
    f.fit()

    # # model test
    # t = Test(path_file="../Dataset/paths.log", output_files='out_put/datasetpredict')
    # t.output()
    
    ## auto simulate bond breaking, predict&truth
    # t = echo(path_file='../Dataset/paths.log', output_files='out_put/datasetpredict')
    # t.output()

    # echo attention score
    # score = Score()
    # score.echo()

    # # protein trainset and testset process
    # database = BuildDatabase(path_file='../protein/train_data/paths.log', dataset_path="dataset/protein/train_data", num_of_cores=16)
    # database.build()
    # database = BuildDatabase(path_file='../protein/test_data/paths.log', dataset_path="dataset/protein/test_data", num_of_cores=16)
    # database.build()

    # model training
    # f = Fit(dataset_path='dataset/traindataset/all_graphs.bin', output_files='out_put/train')
    # f.fit()

    # # model test
    # t = Test(path_file="../protein/test_data/paths.log", output_files='out_put/proteinpredict')
    # t.output()