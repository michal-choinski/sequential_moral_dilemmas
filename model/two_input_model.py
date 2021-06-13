from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf


tf1, tf, tfv = try_import_tf()


class CRPCustomQModel(DistributionalQTFModel):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, conv_filters, fcnet_hiddens, **kw):
        super(CRPCustomQModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        activation = get_activation_fn(
            self.model_config.get("conv_activation"), framework="tf")

        self.inputScreen = tf.keras.layers.Input(
            shape=obs_space.original_space["screen"].shape, name="screen")
        self.inputParameters = tf.keras.layers.Input(
            shape=obs_space.original_space["parameters"].shape, name="parameters")

        self.data_format = "channels_last"

        last_layer = self.inputScreen

        # convolutional layers excluding the last one
        for i, (out_size, kernel, stride) in enumerate(conv_filters[:-1], 1):
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                data_format="channels_last",
                name="conv{}".format(i))(last_layer)

        out_size, kernel, stride = conv_filters[-1]

        # last convolutional layer
        last_layer = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            data_format="channels_last",
            name="conv{}".format(len(conv_filters)))(last_layer)

        last_layer = tf.keras.layers.Flatten()(last_layer)

        self.last_layer_is_flattened = True

        if not self.last_layer_is_flattened:
            last_layer = tf.keras.layers.Lambda(
                lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)

        # combine convolutional layers with additional input features
        last_layer = tf.keras.layers.concatenate([last_layer, self.inputParameters])

        # fully connected layers
        for i in range(len(fcnet_hiddens)):
            last_layer = tf.keras.layers.Dense(
                fcnet_hiddens[i],
                name="hidden{}".format(i),
                activation=tf.nn.relu,
                kernel_initializer=normc_initializer(1.0))(last_layer)

        # output layer
        output_layer = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(last_layer)

        self.base_model = tf.keras.Model(inputs=[self.inputScreen, self.inputParameters], outputs=output_layer)

    def forward(self, input_dict, state, seq_lens):
        screen = input_dict["obs"]["screen"]
        parameters = input_dict["obs"]["parameters"]
        model_out = self.base_model({"screen": screen, "parameters": parameters})
        return model_out, state

    def metrics(self):
        return {"foo": tf.constant(42.0)}