def preprocessing_dense_data(train_directory=train_dir,
                             test_directory=test_dir,
                             ts_tuple=(256,256), 
                             color='grayscale', 
                             batch_size=None,
                             process_test=False):

    
    arg_dict = {'target_size':ts_tuple, 
                'color_mode':color, 
                'batch_size':batch_size}

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
       
    train_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='training') 

    val_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='validation')
    
    
    train_images, train_labels = next(train_generator)
    
    val_images, val_labels = next(val_generator)
    
    
    X_train = train_images.reshape(train_images.shape[0], -1)
    
    X_val = val_images.reshape(val_images.shape[0], -1)
    
    
    y_train = np.reshape(train_labels[:,0], (train_images.shape[0],1))
    
    y_val = np.reshape(val_labels[:,0], (val_images.shape[0],1))
    
    
    return X_train, X_val, y_train, y_val
    
    

    if process_test:
        
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, **arg_dict, subset='test')
        
        test_images, test_labels = next(test_generator)
        
        X_test = test_images.reshape(test_images.shape[0], -1)
        
        y_test = np.reshape(test_labels[:,0], (test_images.shape[0],1))
        
        return X_test, y_test
    

def visualize_nn(history, model, X_train, y_train, X_val, y_val):
    acc = history.history['accuracy']
    mean_acc = np.mean(history.history['accuracy'])
    val_acc = history.history['val_accuracy']
    mean_val_acc = np.mean(history.history['val_accuracy'])
    loss = history.history['loss']
    mean_loss = np.mean(history.history['loss'])
    val_loss = history.history['val_loss']
    mean_val_loss = np.mean(history.history['val_loss'])
    recall = history.history['recall']
    mean_recall = np.mean(history.history['recall'])
    val_recall = history.history['val_recall']
    mean_val_recall = np.mean(history.history['val_recall'])

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and validation recall')
    plt.legend()
    plt.show()
    
    
    print(f'Mean Training Accuracy: {mean_acc}')
    print(f'Mean Validation Accuracy: {mean_val_acc}')
    print('')
    print(f'Mean Training Loss: {mean_loss}')
    print(f'Mean Validation Loss: {mean_val_loss}')
    print('')
    print(f'Mean Training Recall: {mean_recall}')
    print(f'Mean Validation Recall: {mean_val_recall}')
    print('')
    print('Training Evaluation:')
    model.evaluate(X_train, y_train)
    print('')
    print('Validation Evaluation:')
    model.evaluate(X_val, y_val)
                                   
    
    
    preds = (model.predict(X_val) > 0.5).astype('int32')
    
    preds2 = (model.predict(X_train) > 0.5).astype('int32')

    cm = confusion_matrix(y_val, preds)
    
    cm2 = confusion_matrix(y_train, preds2)
    
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    cmd2 = ConfusionMatrixDisplay(cm2, display_labels=['Normal', 'Pneumonia'])
    
    print('Validation Confusion Matrix:')
    cmd.plot()
    print('')
    cmd2.plot()
    print('Training Confusion Matrix:');



def nn_model(dense_list, 
             train_directory=train_dir,
             test_directory=test_dir,
             ts_tuple=(256,256), 
             color='grayscale',
             batch_size=1000,
             process_test=False,
             input_nodes=64,
             input_activation='relu',
             layer_activation='relu',
             output_activation='sigmoid',
             l2_rate=0.01,
             optimizer='sgd',
             loss='binary_crossentropy',
             metrics=['accuracy', 'Recall'],
             epochs=50,
             bs=10):
    
    if not process_test:
        X_train, X_val, y_train, y_val = preprocessing_dense_data(batch_size=batch_size)
    
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = preprocessing_dense_data(batch_size=batch_size)
    
    nn_model = models.Sequential()
    
    
    nn_model.add(layers.Dense(input_nodes, activation=input_activation, input_shape=(X_train.shape[1],)))
    
    for i, val in enumerate(dense_list):
        nn_model.add(layers.Dense(int(dense_list[i]), 
                                  activation=layer_activation,
                                  kernel_regularizer=l2(l2=l2_rate)))
        
        
    nn_model.add(layers.Dense(1, activation=output_activation))
        
    nn_model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=metrics)
    
    
    hist = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_data=(X_val, y_val))
    
    visualize_nn(hist, nn_model, X_train, y_train, X_val, y_val)



def visualize_cnn_test(history, model, train_generator, test_generator):
    

    
    
    acc = history.history['accuracy']
    mean_acc = np.mean(history.history['accuracy'])
    test_acc = history.history['test_accuracy']
    mean_test_acc = np.mean(history.history['test_accuracy'])
    
    loss = history.history['loss']
    mean_loss = np.mean(history.history['loss'])
    test_loss = history.history['test_loss']
    mean_test_loss = np.mean(history.history['test_loss'])
    
    recall = history.history['recall']
    mean_recall = np.mean(history.history['recall'])
    test_recall = history.history['test_recall']
    mean_test_recall = np.mean(history.history['test_recall'])

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, test_acc, 'b', label='Test accuracy')
    plt.title('Training, and Test accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, test_loss, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, test_recall, 'b', label='Test recall')
    plt.title('Training and Test recall')
    plt.legend()
    plt.show()
    
    
    print(f'Mean Training Accuracy: {mean_acc}')
    print(f'Mean Testing Accuracy: {mean_test_acc}')
    print('')
    print(f'Mean Training Loss: {mean_loss}')
    print(f'Mean Testing Loss: {mean_test_loss}')
    print('')
    print(f'Mean Training Recall: {mean_recall}')
    print(f'Mean Testing Recall: {mean_test_recall}')
    print('')
    print('Training Evaluation:')
    model.evaluate(train_generator)
    print('')
    print('Testing Evaluation:')
    model.evaluate(test_generator)
                                   
    
    
    preds = (model.predict(test_generator) > 0.5).astype('int32')                
                
    
    cm = confusion_matrix(test_generator.classes, preds3)
    

    
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    
    cmd.plot();



def visualize_cnn(history, model, train_generator, val_generator):
    
    
    acc = history.history['accuracy']
    mean_acc = np.mean(history.history['accuracy'])
    val_acc = history.history['val_accuracy']
    mean_val_acc = np.mean(history.history['val_accuracy'])
    loss = history.history['loss']
    mean_loss = np.mean(history.history['loss'])
    val_loss = history.history['val_loss']
    mean_val_loss = np.mean(history.history['val_loss'])
    recall = history.history['recall']
    mean_recall = np.mean(history.history['recall'])
    val_recall = history.history['val_recall']
    mean_val_recall = np.mean(history.history['val_recall'])

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and validation recall')
    plt.legend()
    plt.show()
    
    
    print(f'Mean Training Accuracy: {mean_acc}')
    print(f'Mean Validation Accuracy: {mean_val_acc}')
    print('')
    print(f'Mean Training Loss: {mean_loss}')
    print(f'Mean Validation Loss: {mean_val_loss}')
    print('')
    print(f'Mean Training Recall: {mean_recall}')
    print(f'Mean Validation Recall: {mean_val_recall}')
    print('')
    print('Training Evaluation:')
    model.evaluate(train_generator)
    print('')
    print('Validation Evaluation:')
    model.evaluate(val_generator)
                                   
    
    
    
    preds = (model.predict(val_generator) > 0.5).astype('int32') 

    
    cm = confusion_matrix(val_generator.classes, preds)
    
      
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    
    cmd.plot()
    ;


def cnn_model(cnn_list,
              dense_list,
              filters=32,
              kernel_size=(3,3),
              conv_activation='relu',
              input_shape=(256,256,1),
              pool_size=(2,2),
              conv_kernel_size=(3,3),
              conv_layer_activation='relu',
              train_directory=train_dir,
              test_directory=test_dir,
              ts_tuple=(256,256), 
              color='grayscale',
              batch_size=50,
              process_test=False,
              input_activation='relu',
              layer_activation='relu',
              conv_normal=False,
              normal=False,
              output_nodes=1,
              output_activation='sigmoid',
              l2_rate=0.01,
              optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall'],
              steps_per_epoch=60,
              epochs=20,
              bs=10,
              validation_steps=20,
              visualize=True):
    
    if not process_test:
        train_generator, val_generator = cnn_preprocessing(batch_size=batch_size)
    
    else:
        train_generator, val_generator, test_generator = cnn_preprocessing(batch_size=batch_size)
    
    print('ok')
    
    cnn_model = models.Sequential()
    
    cnn_model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=conv_activation,
                            input_shape=input_shape))
    
    
    for i, val in enumerate(cnn_list):
        
        if conv_normal:

            
            cnn_model.add(layers.MaxPooling2D(pool_size))
        
            cnn_model.add(layers.Conv2D(cnn_list[i], kernel_size=conv_kernel_size, use_bias=False))
            
            cnn_model.add(layers.BatchNormalization())
            
            cnn_model.add(layers.Activation(conv_layer_activation))
            
        else:
            
            cnn_model.add(layers.MaxPooling2D(pool_size))
        
            cnn_model.add(layers.Conv2D(cnn_list[i], kernel_size=conv_kernel_size, activation=conv_layer_activation))
    
    
    cnn_model.add(layers.Flatten())
    
    
    for ind, value in enumerate(dense_list):
    
        
        if normal:
            
            cnn_model.add(layers.Dense(dense_list[ind], use_bias=False, kernel_regularizer=l2(l2=l2_rate)))
            
            cnn_model.add(layers.BatchNormalization())
            
            cnn_model.add(Activation(layer_activation))
        
        
        else:
            
            cnn_model.add(layers.Dense(dense_list[ind], activation=layer_activation, kernel_regularizer=l2(l2=l2_rate)))
        
            
    
    cnn_model.add(layers.Dense(output_nodes, activation=output_activation))

    
    cnn_model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)
    print('ok')
    
    hist = cnn_model.fit(train_generator, 
                             steps_per_epoch=steps_per_epoch, 
                             epochs=epochs, 
                             batch_size=bs, 
                             validation_data=(val_generator),
                             validation_steps=validation_steps)
    
    if visualize:
    
        if process_test:
        
            visualize_cnn_test(hist, cnn_model, train_generator, test_generator)
         
        else:
        
            visualize_cnn(hist, cnn_model, train_generator, val_generator)
            
    else:
        
        pass
    
