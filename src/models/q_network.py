import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Neural Network for approximating Q-values."""

    def __init__(self, img_size, action_size, stack_size, seed=42):
        """Initialize parameters and build model.
        Args:
            img_size (tuple): Dimensions of each input image (H, W).
            action_size (int): Dimension of each action.
            stack_size (int): Number of frames stacked to form a state.
            seed (int): Random seed.
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed) # Set seed for this network's initial weights

        self.stack_size = stack_size # Number of input channels (stacked frames)
        height, width = img_size # img_size is expected to be (H, W) e.g. (84,84)

        # Convolutional layers
        # Input: (batch_size, stack_size, height, width)
        self.conv1 = nn.Conv2d(self.stack_size, 32, kernel_size=8, stride=4) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)            # (B, 32, 20, 20) -> (B, 64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)            # (B, 64, 9, 9)   -> (B, 64, 7, 7)

        # Calculate the flattened size after conv layers
        # self.fc_input_size = self._get_conv_output_size((self.stack_size, height, width))
        # For (84,84) input:
        # Conv1: ((84-8)/4)+1 = 19.5 -> 20. Output: (20,20)
        # Conv2: ((20-4)/2)+1 = 8.5 -> 9. Output: (9,9)
        # Conv3: ((9-3)/1)+1 = 7. Output: (7,7)
        # Flattened size: 64 * 7 * 7 = 3136
        self.fc_input_size = 64 * 7 * 7

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, action_size)

        # Initialize weights
        self._initialize_weights()

        print(f"QNetwork initialized with input shape assumption: (batch, {self.stack_size}, {height}, {width})")
        print(f"Action size: {action_size}")
        print(f"Calculated FC input size: {self.fc_input_size}")

    def _initialize_weights(self):
        """Initialize weights of the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _get_conv_output_size(self, shape):
        """Helper function to calculate the output size of convolutional layers."""
        with torch.no_grad():
            x = torch.zeros(1, *shape) # (1, C, H, W)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))

    def forward(self, state):
        """Build a network that maps state -> action values.
        Args:
            state (torch.Tensor): Input state (batch_size, stack_size, height, width).
        Returns:
            torch.Tensor: Q-values for each action (batch_size, action_size).
        """
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from conv layers
        x = x.reshape(x.size(0), -1) # More robust than x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation for the hidden layer
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x) # Output layer (linear activation for Q-values)
        
        return q_values

if __name__ == '__main__':
    # Example Usage and Test
    # These parameters should match those in train.py and agent.py
    # Typically, SHAPE from train.py is (height, width) e.g. (84, 84)
    # STACK_SIZE from train.py is e.g. 4
    # ACTION_SIZE from train.py (via env.action_space.n) is e.g. 5
    # STACK_SIZE from train.py (CONFIG['environment']['stack_size']) is e.g. 4

    IMG_HEIGHT_EXAMPLE = 84
    IMG_WIDTH_EXAMPLE = 84
    ACTION_SIZE_EXAMPLE = 5
    STACK_SIZE_EXAMPLE = 4 # This is now passed to constructor

    # Create a dummy QNetwork instance
    # The QNetwork expects img_size = (height, width) and stack_size
    net = QNetwork(
        img_size=(IMG_HEIGHT_EXAMPLE, IMG_WIDTH_EXAMPLE), 
        action_size=ACTION_SIZE_EXAMPLE,
        stack_size=STACK_SIZE_EXAMPLE,
        seed=123 # Example seed
    )
    print("\nModel Architecture:")
    print(net)

    # Create a dummy input tensor representing a batch of stacked frames
    # Shape: (batch_size, stack_size, height, width)
    batch_size = 2
    dummy_input = torch.randn(batch_size, STACK_SIZE_EXAMPLE, IMG_HEIGHT_EXAMPLE, IMG_WIDTH_EXAMPLE)
    print(f"\nInput tensor shape: {dummy_input.shape}")

    # Perform a forward pass
    try:
        output = net(dummy_input)
        print(f"Output tensor shape: {output.shape}") # Expected: (batch_size, action_size)
        assert output.shape == (batch_size, ACTION_SIZE_EXAMPLE), "Output shape is incorrect!"
        print("\nTest forward pass successful!")
    except Exception as e:
        print(f"\nError during forward pass: {e}")

    # Test _get_conv_output_size (optional, usually for dynamic calculation)
    # calculated_fc_size = net._get_conv_output_size((STACK_SIZE_EXAMPLE, IMG_HEIGHT_EXAMPLE, IMG_WIDTH_EXAMPLE))
    # print(f"Dynamically calculated FC input size: {calculated_fc_size}")
    # assert calculated_fc_size == net.fc_input_size, "Mismatch in FC input size calculation!"
