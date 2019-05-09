import json


def GetLayers(filename):
    ret_list = []
    with open(filename) as f:
        data = json.load(f)

    x = 1
    while x <= len(data):
        layer_data = data["layer" + str(x)]
        layer = {"input_size": -1, "output_size": -1, "perceptrons" : []}
        layer["input_size"] = layer_data["size_in"]
        layer["output_size"] = layer_data["size_out"]
        layer_perceptrons = layer_data["weights"]

        y = 1
        while y <= len(layer_perceptrons):
            layer["perceptrons"].append(layer_perceptrons[str(y)])
            y += 1

        ret_list.append(layer)

        x += 1

    return ret_list


def ProcessLayer(input, output_length, layer):
    output = []
    for x in range(output_length):
        output.append(0.0)

    for perceptron in layer["perceptrons"]:
        for weight in perceptron:
            index = int(weight) - 1
            output[index] += (input[index] * float(perceptron[weight]))

    return output


def Recursion(input, nn_layers, x):
    output = ProcessLayer(input, int(nn_layers[x]["output_size"]), nn_layers[x])

    if x < len(nn_layers) - 1:
        output = Recursion(output, nn_layers, x + 1)

    return output


def NeuralNetwork(filename):
    nn_layers = GetLayers(filename)
    vars = []

    for x in range(int(nn_layers[0]["input_size"])):
        vars.append(1.0)

    return Recursion(vars, nn_layers, 0)


print(NeuralNetwork("data.json"))
