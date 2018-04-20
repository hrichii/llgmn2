import csv

with open("./input_data/lea_sig.csv", "r", encoding="utf-8") as f_csv_input:
    o_csv = csv.reader(f_csv_input)
    l_data=[list(map(float,row)) for row in o_csv]
    