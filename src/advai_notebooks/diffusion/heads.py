"""View of the UNet's native output layer; no additional prediction head."""


class DenoisingHead:
    def __init__(self, unet):
        self.output = unet.conv_out
