import pandas
from defines import Types


def load(metadata):
    print 'Loading data file from location: \'' + metadata.LOCATION + '\''
    # skip first row if CSV is labeled
    skiprows = None
    if metadata.IS_LABELED:
        skiprows = 1
        print 'Skipping header row...'
    # parse CSV file
    data = pandas.read_csv(metadata.LOCATION, header=None, skiprows=skiprows)
    # remove first column if CSV is indexed
    if metadata.IS_INDEXED:
        data = data.iloc[:, 1:]
        print 'Removing index column...'
    # parse data into labels and features
    labels = data.iloc[:, -1]
    features = data.iloc[:, 0:-1]
    # apply feature labels to features
    features.columns = [column.HEADER for column in metadata.COLUMNS]
    print 'Parse complete: ' + str(features.shape[1]) + ' features and ' + str(features.shape[0]),
    print 'samples\n'
    return features, labels

def encode(data, columns):
    print 'Encoding categorical data into numerical values'
    for column in columns:

        # encode numerical data into range-based categories
        if column.TYPE is Types.NUMERICAL:
            data[column.HEADER] = data[column.HEADER].astype(float)

            if column.CATEGORIES is not None:
                data[column.HEADER] = data[column.HEADER].apply(
                    lambda x: _map(column.CATEGORIES, x))
                print 'Column \'' + column.HEADER + '\': numerical data mapped to',
                print 'feature values ' + str([x[1] for x in column.CATEGORIES])

            else:
                headers = data.columns.tolist()
                headers.remove(column.HEADER)
                headers = [column.HEADER] + headers
                data = data[headers]

        # map unordered categories and encode category vectors
        elif column.TYPE is Types.CATEGORICAL_UNORDERED:
            data[column.HEADER] = data[column.HEADER].apply(
                lambda x: column.CATEGORIES[x] if x in column.CATEGORIES else x)
            data[column.HEADER] = data[column.HEADER].astype(float)
            data = pandas.get_dummies(data, columns=[column.HEADER])
            print 'Column \'' + column.HEADER + '\': unordered categorical data mapped to',
            print str(len(set(column.CATEGORIES.values()))) + ' new features'

        # encode ordered and binary categories
        else:
            data[column.HEADER] = data[column.HEADER].apply(
                lambda x: column.CATEGORIES[x] if x in column.CATEGORIES else x)
            data[column.HEADER] = data[column.HEADER].astype(float)
            print 'Column \'' + column.HEADER + '\': categorical data mapped to feature values',
            print str(sorted(set(column.CATEGORIES.values())))

    print 'Encoding complete: ' + str(data.shape[1]) + ' features and ' + str(data.shape[0]) + ' samples\n'
    return data

def _map(checks, x):
    res = [_map_check(check[0], x) for check in checks]
    return checks[res.index(True)][1]

def _map_check(check, x):
    l, r = check.split('_', 1)
    l_sat = [l[0] == '[', l[0] == '(', l[0] == ':']
    r_sat = [r[-1] == ']', r[-1] == ')', r[-1] == ':']
    l_res = True if l_sat[2] else [x >= float(l[1:]), x > float(l[1:])][l_sat.index(True)]
    r_res = True if r_sat[2] else [x <= float(r[:-1]), x < float(r[:-1])][r_sat.index(True)]
    return l_res and r_res
