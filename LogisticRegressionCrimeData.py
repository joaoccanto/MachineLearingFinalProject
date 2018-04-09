def main():
    data_file = 'data_set.txt'
    data_set, label = load_data_set(data_file)
    print len(data_set), len(data_set[0]), len(label)
    i = 0
    #find out how many sets contains -1 or missing data
    for row in data_set:
        if [-1] in row:
            i = i + 1
    print "count, ", i
    
def load_data_set(data_file):
    data_set = []
    label = []
    with open(data_file, 'r') as f:
        for line in f:
            row = []
            sline = line.replace('\n', '')
            t = sline.split(',')
            #load data set
            # first five elements are not predictive and the last element is the label
            for i in range(5, 126):
                # for now missing data points will be recorded as a -1
                if t[i] == '?':
                    row.append([-1])
                else:
                    row.append([float(t[i])])
            data_set.append(row)
            label.append([float(t[127])])
    return data_set, label

main()