import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser("Halp")
parser.add_argument("csv_file", nargs='+', type=str)
parser.add_argument("-o", type=str, default='out.csv.bz2')

parser = parser.parse_args()


data = None
for file_name in parser.csv_file:
    df = pd.read_csv(file_name)
    if data is None:
        data = df
    else:
        data = data.append(df)


data = data.drop_duplicates()
data.to_csv(parser.o)

