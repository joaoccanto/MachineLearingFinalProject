import numpy as np

class LR(object):
    def __init__(self, eta=0.1, conv=.0001):
        self.eta = eta
        self.conv = conv
        self.w = None
    def init_weights(self, num_of_features):
        #self.w = np.random.random_sample((1, num_of_features))
        self.w = np.zeros((1, num_of_features))
        return
    def net_input(self, data):
        return np.dot(data, self.w.T)
    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))
    
    def cost_function(self, data, label):
        z = self.net_input(data)
        y_hat = self.sigmoid(z)
        cost1 = -(label * np.log(y_hat))
        cost0 = (1 - label) * np.log(1 - y_hat)
        cost = cost1 - cost0
        return np.mean(cost)

    def logistic_grad(self, data,labels):
        m = len(labels)
        y_hat = self.sigmoid(self.net_input(data))
        dw = np.dot(data.T, (y_hat - labels))
        return dw/m
    def grad_descent(self, data, label):
        cost = self.cost_function(data, label)
        change_cost = 1
        i = 0
        while change_cost > self.conv:
            old_cost = cost
            dw = self.logistic_grad(data, label)
            self.w -= (self.eta * dw.T)
            cost = self.cost_function(data, label)
            change_cost = old_cost - cost
            print cost
            i = i +1
        print i, change_cost
        return
    def predict(self, data):
        y_hat = self.sigmoid(self.net_input(data))
        num_of_samples = len(data)
        predictions = np.zeros((1,num_of_samples))
        index = 0
        for y in y_hat:
            if(y > .5):
                predictions[0, index] = 1
            index += 1
        return predictions
        

def main():
    data_file = 'data_set.txt'
    data_set, label = load_data_set(data_file)

    data_reduced_sample_space, label_reduced_sample_space = decrease_sample_size(data_set, label)

    data_reduced_feature_space = decrease_feature_space(data_set)

    new_label = lr_label(label_reduced_sample_space)

    logistic_red_sample_size = LR()
    logistic_red_sample_size.init_weights(len(data_reduced_sample_space[0]))
    logistic_red_sample_size.grad_descent(data_reduced_sample_space, new_label)
    predictions = logistic_red_sample_size.predict(data_reduced_sample_space)
    num_of_data = len(data_reduced_sample_space)
    
    #logistic_reduced_features = LR()
    #logistic_reduced_features.init_weights(len(data_reduced_feature_space[0]))
    #logistic_reduced_features.grad_descent(data_reduced_feature_space, new_label)
    #predictions = logistic_reduced_features.predict(data_reduced_feature_space)
    #num_of_data = len(data_reduced_feature_space)

    count = 0
    for p, l in zip(predictions.T, new_label):
        if p == l:
            count = count + 1
    print count, num_of_data

def load_data_set(data_file):
    data_set = []
    label = []
    with open(data_file, 'r') as f:
        for line in f:
            row = np.ones(1)
            sline = line.replace('\n', '')
            t = sline.split(',')
            #load data set
            # first five elements are not predictive and the last element is the label
            for i in range(5, 126):
                # for now missing data points will be recorded as a -1
                if t[i] == '?':
                    row = np.append(row, [-1])
                else:
                    row = np.append(row, [float(t[i])])
            data_set.append(row)
            label.append(float(t[127]))
    return np.asarray(data_set), np.asarray(label)

# remove any sample set that is missing a feature
def decrease_sample_size(data, label):
    reduced_data = []
    reduced_label = []
    for row, lbl in zip(data, label):
        if not ([-1] in row):
            reduced_data.append(row)
            reduced_label.append(lbl)
    return np.asarray(reduced_data), np.asarray(reduced_label)

def decrease_feature_space(data):
    num_of_samples = (len(data))
    num_of_features = len(data[0])
    reduced_data = np.ones((num_of_samples,1))
    incomplete_features = []
    # find out the features missing for at least one set
    for row in data:
        index = 0
        for element in row:
            index = index + 1
            if element == -1 and not (index in incomplete_features):
                incomplete_features.append(index)
    # delete the features
    for i in range(0, num_of_features):
        if not(i in incomplete_features):
            col = data[:,[i]]
            reduced_data = np.hstack((reduced_data, col))

    #return the reduced data 
    return np.delete(reduced_data, np.s_[1:2], axis=1)

def lr_label(label):
    new_label = []
    #if the crime rate is greater than .30, then label the community not safe
    for l in label:
        if l > .20:
            new_label.append([0])
        else:
            new_label.append([1])
    return np.asarray(new_label)
main()