import tensorflow as tf
from prior import MixturePrior
from models import lenet_300_100
from data import get_mnist_data

J = 17
pi0 = 0.999
init_means = tf.linspace(-0.6, 0.6, J - 1)
model = lenet_300_100()
tau = 0.005
lr_w = 1e-4
lr_mu = 1e-3
lr_sigma = 1e-2
lr_pi = 1e-2

prior = MixturePrior(J, pi0, init_means)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_w),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_weights = []
for layer in model.layers:
    if hasattr(layer, "kernel"):
        model_weights.append(layer.kernel)

prior(model_weights)

model_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_w)
mu_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_mu)
sigma_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sigma)
pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_pi)

(x_train, y_train), (x_test, y_test) = get_mnist_data()

with tf.GradientTape(persistent=True) as tape:
    predictions = model(x_train[:100], training=True)
    err_loss = tf.keras.losses.categorical_crossentropy(y_train[:100], predictions)

    model_weights = [layer.kernel for layer in model.layers if hasattr(layer, "kernel")]

    complex_loss = prior.complexity_loss(model_weights)
    total_loss = tf.reduce_mean(err_loss) + tau * complex_loss  # type: ignore

model_grad = tape.gradient(total_loss, model.trainable_variables)
mu_grad = tape.gradient(total_loss, prior.trainable_variables[0])
sigma_grad = tape.gradient(total_loss, prior.trainable_variables[1])
pi_grad = tape.gradient(total_loss, prior.trainable_variables[2])

model_optimizer.apply_gradients(zip(model_grad, model.trainable_variables))  # type: ignore
mu_optimizer.apply_gradients(zip(mu_grad, [prior.trainable_variables[0]]))  # type: ignore
sigma_optimizer.apply_gradients(zip(sigma_grad, [prior.trainable_variables[1]]))  # type: ignore
pi_optimizer.apply_gradients(zip(pi_grad, [prior.trainable_variables[2]]))  # type: ignore
