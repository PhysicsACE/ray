class SortKey:

    def __init__(self, key, descending = False) -> None:
        self.key = []
        self.columns = []
        self.order = []

        if key is not None:

            if isinstance(key, str):
                self.key.append((key, "descending" if descending else "ascending"))
                self.columns.append(key)
                self.order.append(descending)
            
            elif isinstance(key, list):
                
                for k in key:
                    if isinstance(k, str):
                        self.key.append((k, "descending" if descending else "ascending"))
                        self.columns.append(k)
                        self.order.append(descending)
                    elif isinstance(k, tuple):
                        self.key.append(k)
                        self.columns.append(k[0])
                        self.order.append(True if k[1] == "descending" else False)

        self.len = len(self.key)

    
    def to_arrow_sort_args(self):
        return self.key
    
    def to_pandas_sort_args(self):

        def inverse(n):
            return not n
        
        return self.columns, list(map(inverse, self.order))
    
    def to_polars_sort_args(self):
        return self.columns, self.order
    
    def normalized_key(self):
        return self.key
    
    def get_columns(self):
        return self.columns
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        self.curr_index = 0
        return self
    
    def __next__(self):
        curr = self.curr_index

        if curr >= self.len:
            raise StopIteration
        
        self.curr_index += 1
        return curr, self.key[curr]
    
    def __repr__(self):
        builder = "["
        for i in range(self.len):
            builder += "(" + self.key[i][0] + "," + self.key[i][1] + ")"
            if i + 1 != self.len:
                builder += ","
        builder += "]"
        return builder