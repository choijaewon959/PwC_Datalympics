def num_hidden_layer1(num_input, num_output, size):
    # alpha is usually set to be a value to be between 2-10
    alpha = 10
    return size /(num_input + num_output) / alpha
def num_hidden_layer2(num_input, num_output):
    # when single hidden layer
    return 2*((num_output +2)*num_input)**0.5
def num_hidden_layer3(num_input, num_output,size):
    # when two hidden layer
    return [((num_output+2)*num_input)**0.5 + 2*(num_input/(num_output+2))**0.5, num_output*(num_input/(num_output+2))**0.5]

def convert_label(Y):
    '''
    Converting the label into binary vector forms for keras neural network output layer.

    :param: label
    :return: converted label (int vector)
    '''
    l = np.array([[0,0,0,0,0,0,0,0,0,0]])
    tmp = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in Y:
        tmp[int(i)] = 1
        l = np.append(l,[tmp],axis=0)
        tmp = np.array([0,0,0,0,0,0,0,0,0,0])
    l = np.delete(l,0,0)
    YY = pd.DataFrame(l)
    return YY

def add_nodes(self):
    '''
    'dummy' nodes added
    '''
    stop = ['sub_grade','emp_length','loan_status','annual_inc','term','grade', 'delinq_2yrs','inq_last_6mths', 'pub_rec']
    for col in list(self.__transactionData.columns.values):
        try:
            if(stop.index(col) != -1):
                continue
        except:
            if(len(self.__transactionData[col].unique()) < 30):
                for uniq in self.__transactionData[col].unique():
                    self.__transactionData[col+' '+str(uniq)] = self.__transactionData[col].apply(self.additional_feature,args=(uniq,))
    self.__transactionData = self.__transactionData.drop(['home_ownership', 'initial_list_status','application_type'], axis=1)
    print(self.__transactionData.columns.values)
    print(len(self.__transactionData.columns.values))
