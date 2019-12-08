class AsciiTable:
    def __init__(self, table):
        self.table = ''
        for low in table:
            for col in low:
                if type(col) is str:
                    self.table += col + '\t\t'
                else:
                    self.table += str(col) + '\t\t'
            self.table += '\n'
            


