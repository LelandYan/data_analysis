# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/1/20 8:46'

import pandas as pd
import csv

# f = open('ex7.csv')
# reader = csv.reader(f)
# for line in reader:
#     print(line)
lines = list(csv.reader(open('ex7.csv')))
#print(lines)
header,values = lines[0],lines[1:]
data_dict = {h:v for h,v in zip(header,zip(*values))}
#print(data_dict)
xls_file = pd.ExcelFile('data.xls')
table = xls_file.parse('Sheet1')