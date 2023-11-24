import modules
import numpy as np


class Linear(modules.Module):
    def __init__(self, number_input, number_output):
        super().__init__()

        stdv = 1. / np.sqrt(number_input)

        self.weighs = np.random.uniform(-stdv, stdv, size=(number_output, number_input))
        self.biases = np.random.uniform(-stdv, stdv, size=number_output)

        self.gradient_weight = np.zeros(self.weighs)
        self.gradient_bias = np.zeros(self.biases)

    def updateOutput(self, input):
        self.output = input @ self.weighs.T + self.biases
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput @ self.weighs
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradient_weight = np.sum(input[:, None, :] * gradOutput[:, :, None], axis=0)
        self.gradient_bias = np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradient_weight.fill(0)
        self.gradient_bias.fill(0)

    def getParameters(self):
        return [self.weighs, self.biases]

    def getGradParameters(self):
        return [self.gradient_weight, self.gradient_bias]

    def __repr__(self):
        s = self.weighs.shape
        q = 'Linear %d -> %d' %(s[1], s[0])

        return q


class SoftMax(modules.Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims= True))
        self.output = np.exp(self.output)
        self.output = self.output/np.sum(self.output,axis=1, keepdims=True)

        return self.output

    def updateGradInput(self, input, gradOutput):
        ###
        return self.gradInput

    def __repr__(self):
        return 'SoftMax'
