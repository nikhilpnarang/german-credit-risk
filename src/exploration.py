import pandas
import preprocessing
from defines import Types, Metadata


def print_pivot_tables(data, metadata, numerical=False):
    for column in metadata.COLUMNS:
        if not numerical and column.TYPE is Types.NUMERICAL or column.CATEGORIES is None:
            continue
        df_column = pandas.DataFrame(data[column.HEADER])
        count = df_column.apply(pandas.value_counts).T
        count = count.reindex_axis(sorted(count.columns), axis=1)
        print count, '\n'

def print_statistics(data, metadata):
    for column in metadata.COLUMNS:
        if column.TYPE is not Types.NUMERICAL or column.CATEGORIES is not None:
            continue
        print pandas.DataFrame(data[column.HEADER]).describe(), '\n'

if __name__ == '__main__':
    metadata = Metadata()
    data, _ = preprocessing.load(metadata)
    print_pivot_tables(data, metadata, numerical=True)
    print_statistics(data, metadata)
