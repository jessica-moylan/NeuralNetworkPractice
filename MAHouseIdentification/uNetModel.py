import torch #imports torch library
import torch.nn as nn
import torchvision.transforms.functional as TF

"""
    When looking at the model for how a UNET architecture is set up, we see that for all of the "steps" the image gets 
    'transformed' two times. This is called a convolution.
        CONVOLUTION: a multiplication that is performed between an array of input data and a two-dimensional array of weights

    This is the function that creates the double convolution that brings, as describes in the uNet_arctutrecture png that moves the information, in one group
    from the left to the right.

    nn.Conv2d((in_channels, out_channels, kernel_size, stride, padding, bias)
        - in_channels: the number of channels in the input image, (colored images are 3, black and white is 1(only color in 1 dimension))
        - out_channels: number of channels produced by the convolution
        - kernel_size: widthxheight of the mask (this moves over the image) an int means that the kernal matrix is a nxn matrix and a tuple means a nxm
        - stride: how far the kernal moves each time is does, 1 mean it moves over to the next pixel
        - padding: a 1 means "same" the input hight and width will be the smae after the covolution
        - bias: do we want a learnable bias (do we want to keep the same value throughout the model?)
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), #batchnorm cancels out the bias from conv2d so we can prevent a "unless" parameter by setting bias = False
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    """
        This is the function that actually builds the movment up and down that is needed for analysing and doing the Neural Network stuff that we want
    """
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()

        #we want to be able to do model.eval and for the batch normal layers so that is why we choe nn.ModuleList
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) #matches some input to the next size ---> the first time this ever runs it matches 3 to 64
            in_channels = feature #this changes the number of inputs so we have a 1 --> 64 --> 128 situtaion so the model looks correct

        # Up part of UNET
        for feature in reversed(features):
            #this is the movement from left to right via the gray arrow in which the in_channels on the right side are double the size of the number of features on the left
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2, #the kernel_size being 2 doubles the size (heigh and width) of the image
                )
            )
            self.ups.append(DoubleConv(feature*2, feature)) #this because in each "group" we move over 2 and up one
       
        #bottleneck (the turning point from down to up)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #final convolution on the top right which keeps the same size of the image but decreaes the out_channels which would provide our final predicted result
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x) #the ordering is important here as we move sideways from lowest resolution first than to highest resultion

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2): # up than double conv
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] #we have a step of 2 and we want a linear step of 1 ordering

            #if the two are not the same size, this is important since otherwise it will not work, we will take out height and width
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) #double convolutopm

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    print(preds.shape)
    print(x.shape)

if __name__ == "__main__":
    test()