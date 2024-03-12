import tensorflow as tf

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Prepare your dataset (assuming you have X_train, y_train, X_val, y_val)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Define the number of steps to compute gradients
compute_gradients_every_n_steps = 10

# Custom training loop
for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass
            logits = model(x_batch_train)
            loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_batch_train, logits, from_logits=True)
        
        # Compute gradients every n steps
        if step % compute_gradients_every_n_steps == 0:
            gradients = tape.gradient(loss_value, model.trainable_variables)
            # Apply gradients
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print the loss every n steps
        if step % compute_gradients_every_n_steps == 0:
            print("Step:", step, "Loss:", loss_value.numpy())
