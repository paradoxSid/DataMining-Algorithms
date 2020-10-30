def readFile(fname, split_str=' '):
    """Reads the file with a given name line by line and
    return it as a list of items
    """
    items = []
    types = []

    with open(fname, 'r') as file:
        for line in file.readlines():
            if len(line.strip()) == 0 or line.strip()[0] in ['%', '@']:
                continue
            features = line.strip().split(split_str)
            items.append([float(i) for i in features[:-1]])
            types.append(features[-1])

    return items, types
