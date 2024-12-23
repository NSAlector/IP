from  .filter import Filter
class ComplexFilter(object):
    def __init__(self):
        self.filter_list = []

    def add_filter(self, filter: Filter):
        self.filter_list.append(filter)

    def processing(self, data):
        for p in self.filter_list:
            data = p.processing(data)
        return data
