import numpy as np
import tensorflow as tf

from .layers import Base


class StochDepth(tf.keras.Model):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, x, training):
        if not training:
            return x

        batch_size = tf.shape(x)[0]
        r = tf.random.uniform(shape=[batch_size, 1, 1], dtype=x.dtype)
        keep_prob = tf.cast(1.0 - self.drop_rate, dtype=self._compute_dtype)
        binary_tensor = tf.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class Mlp(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="gelu",
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.act_layer = act_layer
        self.drop_rate = drop_rate

        self.fc1 = tf.keras.layers.Dense(self.hidden_features)
        self.act = tf.keras.layers.Activation(self.act_layer)
        self.fc2 = tf.keras.layers.Dense(self.out_features)
        self.drop = tf.keras.layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training)
        x = self.fc2(x)
        x = self.drop(x, training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"in_features": self.in_features})
        config.update({"hidden_features": self.hidden_features})
        config.update({"out_features": self.out_features})
        config.update({"act_layer": self.act_layer})
        config.update({"drop_rate": self.drop_rate})
        return config


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    input_shape = tf.shape(x)
    B = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = input_shape[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, (-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = tf.shape(windows)[0] // (H * W // (window_size * window_size))

    x = tf.reshape(
        windows, (B, H // window_size, W // window_size, window_size, window_size, -1)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (B, H, W, -1))
    return x


def log_n(x, n):
    return tf.math.log(x) / tf.math.log(tf.cast(n, x.dtype))


class WindowAttention(tf.keras.layers.Layer):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_max = tf.cast(tf.math.log(1.0 / 0.01), dtype=self._compute_dtype)

        logit_init = tf.keras.initializers.Constant(tf.math.log(10.0))
        self.logit_scale = self.add_weight(
            name=f"{self.name}/logit_scale",
            shape=(num_heads, 1, 1),
            initializer=logit_init,
            trainable=True,
        )

        # mlp to generate continuous relative position bias
        self.cpb_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, use_bias=True),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dense(num_heads, use_bias=False),
            ]
        )

        relative_coords_h = range(-(self.window_size[0] - 1), self.window_size[0])
        relative_coords_w = range(-(self.window_size[1] - 1), self.window_size[1])

        relative_coords_table = tf.meshgrid(
            relative_coords_h, relative_coords_w, indexing="ij"
        )
        relative_coords_table = tf.stack(relative_coords_table)
        relative_coords_table = tf.transpose(relative_coords_table, (1, 2, 0))
        relative_coords_table = tf.expand_dims(relative_coords_table, axis=0)
        relative_coords_table = tf.cast(relative_coords_table, self.dtype)

        if self.pretrained_window_size[0] > 0:
            relative_coords_table_0 = relative_coords_table[:, :, :, 0] / (
                self.pretrained_window_size[0] - 1
            )
            relative_coords_table_1 = relative_coords_table[:, :, :, 1] / (
                self.pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table_0 = relative_coords_table[:, :, :, 0] / (
                self.window_size[0] - 1
            )
            relative_coords_table_1 = relative_coords_table[:, :, :, 1] / (
                self.window_size[1] - 1
            )
        relative_coords_table = tf.stack(
            [relative_coords_table_0, relative_coords_table_1], axis=3
        )

        relative_coords_table *= 8
        relative_coords_table = (
            tf.math.sign(relative_coords_table)
            * log_n(tf.math.abs(relative_coords_table) + 1.0, 2)
            / np.log2(8)
        )
        self.relative_coords_table = relative_coords_table

        # get pair-wise relative position index for each token inside the window
        coords_h = range(self.window_size[0])
        coords_w = range(self.window_size[1])

        # 2, Wh, Ww
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing="ij"))

        # 2, Wh*Ww
        coords_flatten = tf.reshape(
            coords, (2, self.window_size[0] * self.window_size[1])
        )

        # 2, Wh*Ww, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Wh*Ww, Wh*Ww, 2
        relative_coords = tf.transpose(relative_coords, (1, 2, 0))

        # shift to start from 0
        relative_coords_0 = relative_coords[:, :, 0] + (self.window_size[0] - 1)
        relative_coords_1 = relative_coords[:, :, 1] + (self.window_size[1] - 1)
        relative_coords_0 = relative_coords_0 * (2 * self.window_size[1] - 1)
        relative_coords = tf.stack([relative_coords_0, relative_coords_1], axis=2)

        # Wh*Ww, Wh*Ww
        relative_position_index = tf.math.reduce_sum(relative_coords, axis=-1)
        self.relative_position_index = relative_position_index

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        if qkv_bias:
            self.q_bias = self.add_weight(
                name=f"{self.name}/q_bias", shape=(dim,), initializer="zeros"
            )
            self.v_bias = self.add_weight(
                name=f"{self.name}/v_bias", shape=(dim,), initializer="zeros"
            )
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, use_bias=True)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x, training=None, mask=None):
        input_shape = tf.shape(x)
        B_, N, C = input_shape[0], input_shape[1], input_shape[2]

        qkv = self.qkv(x)

        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        qkv_bias = None
        if self.q_bias is not None:
            q = tf.math.add(q, tf.reshape(self.q_bias, (self.num_heads, 1, -1)))
            v = tf.math.add(v, tf.reshape(self.v_bias, (self.num_heads, 1, -1)))

        # cosine attention
        attn = tf.math.l2_normalize(q, axis=-1) @ tf.transpose(
            tf.math.l2_normalize(k, axis=-1), (0, 1, 3, 2)
        )
        logit_scale = tf.math.exp(tf.math.minimum(self.logit_scale, self.logit_max))
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table)
        relative_position_bias_table = tf.reshape(
            relative_position_bias_table, (-1, self.num_heads)
        )

        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.reshape(
            tf.gather(
                relative_position_bias_table,
                tf.reshape(self.relative_position_index, (-1,)),
            ),
            (
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ),
        )

        # nH, Wh*Ww, Wh*Ww
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = 16 * tf.math.sigmoid(relative_position_bias)
        attn = attn + tf.expand_dims(relative_position_bias, 0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 0)
            attn = tf.reshape(attn, (B_ // nW, nW, self.num_heads, N, N)) + mask
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn, training=training)

        x = tf.reshape(tf.transpose(attn @ v, (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input reslution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (str, optional): Activation layer. Default: "gelu"
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=tf.keras.layers.LayerNormalization,
        pretrained_window_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size),
        )

        self.drop_path = (
            StochDepth(drop_path, scale_by_keep=True)
            if drop_path > 0.0
            else tf.keras.layers.Layer()
        )
        self.norm2 = norm_layer(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_rate=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            h_sizes = (
                H - self.window_size,
                self.window_size - self.shift_size,
                self.shift_size,
            )
            w_sizes = (
                W - self.window_size,
                self.window_size - self.shift_size,
                self.shift_size,
            )
            cnt = 0
            img_mask = []
            for h in h_sizes:
                img_line = []
                for w in w_sizes:
                    img_line.append(
                        tf.constant(cnt, shape=(h, w), dtype=self._compute_dtype)
                    )
                    cnt += 1
                img_mask.append(tf.concat(img_line, axis=1))

            img_mask = tf.concat(img_mask, axis=0)
            img_mask = tf.expand_dims(tf.expand_dims(img_mask, 0), 3)

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, (-1, self.window_size * self.window_size)
            )
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
                mask_windows, 2
            )
            attn_mask = tf.where(attn_mask != 0, float(-100.0), attn_mask)
            attn_mask = tf.where(attn_mask == 0, float(0.0), attn_mask)
        else:
            attn_mask = None

        self.attn_mask = attn_mask

    def call(self, x, training=None):
        H, W = self.input_resolution
        input_shape = tf.shape(x)
        B = input_shape[0]
        L = input_shape[1]
        C = input_shape[2]

        shortcut = x
        x = tf.reshape(x, (B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = tf.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )

        # B H' W' C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))
        x_normed = self.norm1(x)
        x_dropped = self.drop_path(x_normed, training=training)
        x = shortcut + x_dropped

        # FFN
        x_mlp = self.norm2(self.mlp(x, training=training))
        x = x + self.drop_path(x_mlp, training=training)

        return x


class PatchMerging(tf.keras.layers.Layer):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer.  Default: tf.keras.layers.LayerNormalization
    """

    def __init__(
        self,
        input_resolution,
        dim,
        norm_layer=tf.keras.layers.LayerNormalization,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)
        self.norm = norm_layer(epsilon=1e-5)

    def call(self, x, training=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        input_shape = tf.shape(x)
        B = input_shape[0]
        L = input_shape[1]
        C = input_shape[2]

        x = tf.reshape(x, (B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = tf.reshape(x, (B, int(H / 2 * W / 2), 4 * C))  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


def BasicLayer(
    x,
    dim,
    input_resolution,
    depth,
    num_heads,
    window_size,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
    norm_layer=tf.keras.layers.LayerNormalization,
    downsample=None,
    pretrained_window_size=0,
):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: tf.keras.layers.LayerNormalization
        downsample (tf.keras.layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    # build blocks
    for i in range(depth):
        x = SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer,
            pretrained_window_size=pretrained_window_size,
        )(x)

    # patch merging layer
    if downsample is not None:
        x = downsample(input_resolution, dim=dim, norm_layer=norm_layer)(x)
    return x


def PatchEmbed(x, img_size=224, patch_size=4, embed_dim=96, norm_layer=None):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (tf.keras.layers.Layer, optional): Normalization layer. Default: None
    """

    patch_size = (patch_size, patch_size)

    x = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(x)
    x = tf.keras.layers.Reshape(target_shape=(-1, embed_dim))(x)
    if norm_layer is not None:
        x = norm_layer(epsilon=1e-5)(x)

    return x


def SwinTransformerV2(
    x,
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=tf.keras.layers.LayerNormalization,
    patch_norm=True,
    pretrained_window_sizes=[0, 0, 0, 0],
):
    r"""Swin Transformer
        A TensorFlow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (tf.keras.layers.Layer): Normalization layer. Default: tf.keras.layers.LayerNormalization.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    num_layers = len(depths)

    patch_resolution = img_size // patch_size
    patches_resolution = [patch_resolution, patch_resolution]

    # stochastic depth
    dpr = [x for x in tf.linspace(float(0.0), drop_path_rate, sum(depths))]

    x = PatchEmbed(
        x,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        norm_layer=norm_layer if patch_norm else None,
    )

    x = tf.keras.layers.Dropout(rate=drop_rate)(x)

    # build layers
    for i_layer in range(num_layers):
        x = BasicLayer(
            x,
            dim=int(embed_dim * 2**i_layer),
            input_resolution=(
                patches_resolution[0] // (2**i_layer),
                patches_resolution[1] // (2**i_layer),
            ),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer < num_layers - 1) else None,
            pretrained_window_size=pretrained_window_sizes[i_layer],
        )

    x = norm_layer(epsilon=1e-5, name="predictions_norm")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="predictions_globalavgpooling")(x)

    if num_classes > 0:
        x = tf.keras.layers.Dense(num_classes, name="predictions_dense")(x)
    return x


definitions = {
    "Base": {
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "num_heads": [4, 8, 16, 32],
    }
}


def SwinV2(
    in_shape=(320, 320, 3),
    out_classes=2000,
    definition_name="Base",
    input_scaling="inception",
    window_size=10,
    stochdepth_rate=0.2,
    **kwargs,
):
    definition = definitions[definition_name]
    embed_dim = definition["embed_dim"]
    depths = definition["depths"]
    num_heads = definition["num_heads"]

    img_input = tf.keras.layers.Input(shape=in_shape)
    x = Base.input_scaling(method=input_scaling)(img_input)

    x = SwinTransformerV2(
        x,
        img_size=in_shape[0],
        in_chans=in_shape[2],
        num_classes=out_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        drop_path_rate=stochdepth_rate,
        **kwargs,
    )

    x = tf.keras.layers.Activation("sigmoid", name="predictions_sigmoid")(x)

    model = tf.keras.Model(img_input, x, name=f"SwinV2-{definition_name}")
    return model
