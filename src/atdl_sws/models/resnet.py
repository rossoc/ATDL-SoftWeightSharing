import tensorflow as tf
from keras import layers, Model


class BasicBlock(layers.Layer):
    def __init__(self, in_planes, out_planes, stride, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            out_planes,
            3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            out_planes,
            3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )

        # Shortcut connection
        if stride != 1 or in_planes != out_planes:
            self.shortcut = layers.Conv2D(
                out_planes,
                1,
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
            )
        else:
            self.shortcut = lambda x: x  # Identity

    def build(self, input_shape):
        # Build all sublayers
        super(BasicBlock, self).build(input_shape)

    def call(self, x):
        # Pre-activation structure: BN -> ReLU -> Conv
        out = self.bn1(x)
        out = tf.nn.relu(out)

        residual = (
            self.shortcut(out)
            if self.stride != 1 or self.in_planes != self.out_planes
            else x
        )

        out = self.conv1(out)
        out = self.bn2(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)

        return out + residual

    def get_config(self):
        config = super(BasicBlock, self).get_config()
        config.update(
            {
                "in_planes": self.in_planes,
                "out_planes": self.out_planes,
                "stride": self.stride,
            }
        )
        return config


class NetworkBlock(layers.Layer):
    def __init__(self, n_layers, in_planes, out_planes, stride, **kwargs):
        super(NetworkBlock, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.blocks = []
        for i in range(n_layers):
            block_stride = stride if i == 0 else 1
            block_in_planes = in_planes if i == 0 else out_planes
            self.blocks.append(BasicBlock(block_in_planes, out_planes, block_stride))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_config(self):
        config = super(NetworkBlock, self).get_config()
        config.update(
            {
                "n_layers": self.n_layers,
                "in_planes": self.in_planes,
                "out_planes": self.out_planes,
                "stride": self.stride,
            }
        )
        return config


class ResNet(Model):
    def __init__(self, num_classes=10, k=4, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.k = k

        n = (16 - 4) // 6  # 2 blocks per stage
        self.widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = layers.Conv2D(
            self.widths[0],
            3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.block1 = NetworkBlock(n, self.widths[0], self.widths[1], 1)
        self.block2 = NetworkBlock(n, self.widths[1], self.widths[2], 2)
        self.block3 = NetworkBlock(n, self.widths[2], self.widths[3], 2)
        self.bn = layers.BatchNormalization()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(
            num_classes, kernel_initializer="he_normal", bias_initializer="zeros"
        )

    def call(self, x):
        out = self.conv1(x)  # No BN/ReLU before first block (as in original)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = tf.nn.relu(self.bn(out))
        out = self.avg_pool(out)
        return tf.nn.softmax(self.fc(out))

    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({"num_classes": self.num_classes, "k": self.k})
        return config
