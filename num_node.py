def num_hidden_layer1(num_input, num_output, size):
    # alpha is usually set to be a value to be between 2-10
    alpha = 2
    return size /(num_input + num_output) / alpha
def num_hidden_layer2(num_input, num_output):
    # when single hidden layer
    return 2*((num_output +2)*num_input)**0.5
def num_hidden_layer3(num_input, num_output,size):
    # when two hidden layer
    return [((num_output+2)*num_input)**0.5 + 2*(num_input/(num_output+2))**0.5, num_output*(num_input/(num_output+2))**0.5]
