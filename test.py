import numpy as np
from util.Stat import *
from util.Distribution import Distribution
from data.FileManager import FileManager
from data.Column import Column
from data.Row import Row

x = [0.3818, 0.4909, 0.1227, 0.0045]
y = [0.2545, 0.5091, 0.2182, 0.0182]

X1 = Distribution(x)
Y1 = Distribution(y)

print( covariance_2d(X1, X1) )
print( variance(X1) )

print ( correlation_2d(X1, Y1) )

fm = FileManager("test")
print (fm.retrieve_data())
print (fm.get_row_size())
print (fm.get_col_size())
