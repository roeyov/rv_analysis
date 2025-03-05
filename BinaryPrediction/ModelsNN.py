import torch.nn as nn

class MultiOutputNN(nn.Module):
    def __init__(self):
        super(MultiOutputNN, self).__init__()
        self.layer1 = nn.Linear(78, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.output_is_bianry = nn.Linear(64, 1)
        self.output_num_of_ps = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        float_output = self.output_num_of_ps(x)
        # Compute the binary output and apply sigmoid and threshold
        binary_output = self.sigmoid(self.output_is_bianry(x))
        return binary_output, float_output

