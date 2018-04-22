from __future__ import division

import numpy as np
import sys


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
        print "training accuracy\niteration,cost,correct,total,ratio"
        while change_cost > self.conv:
            old_cost = cost
            dw = self.logistic_grad(data, label)
            self.w -= (self.eta * dw.T)
            cost = self.cost_function(data, label)
            change_cost = old_cost - cost
            predictions = self.predict(data)
            #print cost
            i = i +1

            
            count = 0
            for p, l in zip(predictions.T, label):
                if p == l:
                    count += 1
            print "{},{},{},{},{}".format(i,cost,count, len(data),count/len(data))
        print
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
        
        
def print_results(heading, predictions, label, cities, num_of_data):
    count = 0
    print heading+"\ncity,prediction,truth,correct"
    for p, l, c in zip(predictions.T, label, cities):
        print "{},{},{},{}".format(c,p[0],l[0],p[0]==l[0])
        if p == l:
            count = count + 1
            
    print "\nAccuracy\ncorrect,total,ratio\n{},{},{}\n".format(count,num_of_data,count/num_of_data)
    return

def main():

    sys.stdout = open('file.csv', 'w')

    data_file = 'data_set.txt'
    data_set, label, cities = load_data_set(data_file)

    data_reduced_sample_space, label_reduced_sample_space, cities_reduced_samples = decrease_sample_size(data_set, label,cities)

    new_label = lr_label(label_reduced_sample_space)

    tr_data, tr_label, tr_city = training_sets(data_reduced_sample_space, new_label, cities_reduced_samples)
    t_data, t_label, t_cities = test_sets(data_reduced_sample_space, new_label, cities_reduced_samples)

    logistic_red_sample_size = LR()
    logistic_red_sample_size.init_weights(len(tr_data[0]))
    logistic_red_sample_size.grad_descent(tr_data, tr_label)
    predictions = logistic_red_sample_size.predict(t_data)
    num_of_data = len(t_data)

#    count = 0

#    print "reduced sample space"
#    for p, l, c in zip(predictions.T, t_label, t_cities):
#        print c , "\t" ,p[0] , "\t", l[0]
#        if p == l:
#            count = count + 1
#    print count, num_of_data
    print_results("reduced sample space", predictions, t_label, t_cities, num_of_data)
###########################################################
    data_reduced_feature_space = decrease_feature_space(data_set)
    new_label = lr_label(label)

    tr_data, tr_label, tr_city = training_sets(data_reduced_feature_space, new_label, cities)
    t_data, t_label, t_city = test_sets(data_reduced_feature_space, new_label, cities)

    
    logistic_red_sample_size = LR()
    logistic_red_sample_size.init_weights(len(tr_data[0]))
    logistic_red_sample_size.grad_descent(tr_data, tr_label)
    predictions = logistic_red_sample_size.predict(t_data)
    num_of_data = len(t_data)

#    count = 0

#    print "reduced feature space"
#    for p, l, c in zip(predictions.T, t_label, t_city):
#        print c , "\t" ,p[0] , "\t", l[0]
#        if p == l:
#            count = count + 1
#    print count, num_of_data

    print_results("reduced feature space", predictions, t_label, t_city, num_of_data)
    ############################
    double_reduced_data= decrease_feature_space(data_reduced_feature_space)
    double_reduced_data, double_reduced_label, double_reduced_cities= decrease_sample_size(double_reduced_data, label, cities)
    new_label = lr_label(double_reduced_label)

    tr_data, tr_label, tr_city = training_sets(double_reduced_data, new_label, double_reduced_cities)
    t_data, t_label, t_city = test_sets(double_reduced_data, new_label, double_reduced_cities)

    logistic_red_sample_size = LR()
    logistic_red_sample_size.init_weights(len(tr_data[0]))
    logistic_red_sample_size.grad_descent(tr_data, tr_label)
    predictions = logistic_red_sample_size.predict(t_data)
    num_of_data = len(t_data)

#    count = 0

#    print "reduced sample and feature space"
#    for p, l, c in zip(predictions.T, t_label, t_city):
#        print c , "\t" ,p[0] , "\t", l[0]
#        if p == l:
#            count = count + 1
#    print count, num_of_data
    
    print_results("reduced sample and feature space", predictions, t_label, t_city, num_of_data)

def load_data_set(data_file):
    data_set = []
    label = []
    cities = []
    with open(data_file, 'r') as f:
        for line in f:
            row = np.ones(1)
            sline = line.replace('\n', '')
            t = sline.split(',')
            #load city
            cities.append(t[3])
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
    return np.asarray(data_set), np.asarray(label), cities

# remove any sample set that is missing a feature
def decrease_sample_size(data, label, cities):
    reduced_data = []
    reduced_label = []
    reduced_cities = []
    for row, lbl, c in zip(data, label, cities):
        if not ([-1] in row):
            reduced_data.append(row)
            reduced_label.append(lbl)
            reduced_cities.append(c)
    return np.asarray(reduced_data), np.asarray(reduced_label), reduced_cities

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
    threshold = np.mean(label)
    #if the crime rate is greater than .30, then label the community not safe
    for l in label:
        if l > threshold:
            new_label.append([0])
        else:
            new_label.append([1])
    return np.asarray(new_label)

def training_sets(data, label, city):
    t_data = []
    t_label = []
    t_cities = []
    num_of_data = len(data)

    for i in range(0, (num_of_data - num_of_data // 10)):
        t_data.append(data[i])
        t_label.append(label[i])
        t_cities.append(city[i])
    return np.asarray(t_data), np.asarray(t_label), t_cities

def test_sets(data, label, city):
    t_data = []
    t_label = []
    t_cities = []
    num_of_data = len(data)
    for i in range(num_of_data - num_of_data // 10, num_of_data):
        t_data.append(data[i])
        t_label.append(label[i])
        t_cities.append(city[i])
    return np.asarray(t_data), np.asarray(t_label), t_cities

main()
