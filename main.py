# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : main.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: The program of calling other functions, such as fit,predict

from data.build_dataset import BuildDatabase
from models.fit import Fit
from models.predict import Test

if __name__ == '__main__':
    # database = BuildDatabase(path_file='../Dataset/paths.log', dataset_path="dataset/Dataset", num_of_cores=32)
    # database.build()
    # f = Fit(dataset_path='dataset/Dataset/all_graphs.bin', output_files='out_put/train')
    # f.fit()

    # database = BuildDatabase(path_file='../Testset/paths.log', dataset_path="dataset/Testset", num_of_cores=8)
    # database.build()
    # t = Test(dataset_path='dataset/Testset/all_graphs.bin', output_files='out_put/test')
    # t.test()

    t = Test(path_file='../Dataset/paths.log', output_files='out_put/datasetpredict')
    t.output()



    # # database = BuildDatabase(path_file='../polyextra/CH3CHCHCHCHCHCHCH3/paths.log',
    # #                          dataset_path="dataset/CH3CHCHCHCHCHCHCH3", num_of_cores=8)
    # # database.build()
    # # t = Test(dataset_path='dataset/CH3CHCHCHCHCHCHCH3/all_graphs.bin',
    # #          output_files='out_put/CH3CHCHCHCHCHCHCH3')
    # # t.test()
    #
    # molecule_id = 'CH3CHCHCHCHCHCHCH3_3'
    # database = BuildDatabase(path_file=f'../polyextra/CH3CHCHCHCHCHCHCH3Self/{molecule_id}/X/paths.log',
    #                          dataset_path=f"dataset/CH3CHCHCHCHCHCHCH3Self/{molecule_id}", num_of_cores=4)
    # database.build()
    # t = Test(dataset_path=f'dataset/CH3CHCHCHCHCHCHCH3Self/{molecule_id}/all_graphs.bin',
    #          output_files=f'out_put/CH3CHCHCHCHCHCHCH3Self/{molecule_id}')
    # t.test()

