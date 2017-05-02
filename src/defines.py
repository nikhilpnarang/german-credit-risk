from enum import enum
import json

Types = enum('NUMERICAL',
             'CATEGORICAL_BINARY',
             'CATEGORICAL_ORDERED',
             'CATEGORICAL_UNORDERED')

class Column(object):
    def __init__(self, attribute=None, header='', type=None, categories=None):
        self.ATTRIBUTE = attribute
        self.HEADER = header
        self.TYPE = type
        self.CATEGORIES = categories

class Metadata(object):
    def __init__(self):
        metadata = json.load(open('metadata.json', 'r'))
        self.NAME = metadata['NAME']
        self.LOCATION = metadata['LOCATION']
        self.IS_INDEXED = metadata['IS_INDEXED']
        self.IS_LABELED = metadata['IS_LABELED']
        self.COLUMNS = []
        for column in metadata['COLUMNS']:
            self.COLUMNS.append(Column(attribute=column['attribute'],
                                       header=column['header'],
                                       type=getattr(Types, column['type']),
                                       categories=column['categories']))
