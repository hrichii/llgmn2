import numpy
import math
from matplotlib import pyplot
import csv

csv_file = open("./TEST_STOCK.csv", "r", encoding="ms932", errors="", newline="" )
#リスト形式
A = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
#辞書形式
B = csv.DictReader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)