# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:08:41 2023

@author: ZHANG Jun
"""
#===================================================================================================
#### reference from DOI:https://doi.org/10.1016/j.joule.2023.06.003
#### reference code from url:https://github.com/jzhang-github/AGAT
###  cite:Zhang, Jun & Wang, Chaohui & Huang, Shasha & Xiang, Xuepeng & Xiong, Yaoxu & Biao, Xu
###  & Ma, Shihua & Fu, Haijun & Kai, Jijung & Kang, Xiongwu & Zhao, Shijun. (2023).
###  Design high-entropy electrocatalyst via interpretable deep graph attention learning.
###  Joule. 7. 10.1016/j.joule.2023.06.003.
#===================================================================================================

import os
class FileExit(Exception):
    pass

def file_exit():
    if os.path.exists('StopPython'):
        os.remove('StopPython')
        raise FileExit('Exit because `StopPython` file is found.')

