import tensorflow as tf

def make_model(num_classes, base_model='DenseNet121'):
    if base_model == 'DenseNet121':
        base = tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    elif base_model == 'DenseNet201':
        base = tf.keras.applications.DenseNet201(weights=None, include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported base_model")

    base.trainable = True
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    return model