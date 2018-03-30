
def extract_data(path, clazz):

    with open(path) as f:
        content = f.readlines()

    # Remove whitespaces and \n
    content = [x.strip() for x in content]

    data = []
    for line in content:
        data.append(clazz(line))

    return data
