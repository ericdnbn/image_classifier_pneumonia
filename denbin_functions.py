def preprocessing_dense_data(train_directory=train_dir,
                             test_directory=test_dir,
                             ts_tuple=(256,256), 
                             color='grayscale', 
                             batch_size=None,
                             process_test=False):

'''
Arguments:

This function takes in a training and testing directory, 
a tuple indicating how to resize the image, the color scale, 
the number of images to pull from the directory, 
and a boolean for process_test, which tells the function whether or not to create
a test generator to pull images from the testing directory.

Functionality:

This function creates image generators with the flow_from_directory method called off 
of them to pull images from their respective directories. 
It then splits images pulled by each generator into variables and labels.
The variables and labels are then separately reshaped to be in a single column.
'''

    
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

'''
Arguments:

This function takes in model history, the model itself, X_train, y_train, X_val, and y_val.

Functionality:

This function calculates accuracy, validation accuracy, loss, validation loss, 
recall, and validation recall. It then plots these metrics, 
evaluates the model on the training data, evaluates the model on the validation data,
before finally plotting a confusion matrix showing the results 
of using the model to predict the validation data.

'''

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    recall = history.history['recall']
    val_recall = history.history['val_recall']

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
    

    print('')
    print('Training Evaluation:')
    model.evaluate(X_train, y_train)
    print('')
    print('Validation Evaluation:')
    model.evaluate(X_val, y_val)
                                   
    
    
    preds = (model.predict(X_val) > 0.5).astype('int32')
    

    cm = confusion_matrix(y_val, preds)
    
    
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    
    print('')
    cmd.plot()
    print('')




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



def cnn_preprocessing(train_directory=train_dir,
                      test_directory=test_dir,
                      ts_tuple=(256,256), 
                      color='grayscale', 
                      batch_size=None,
                      process_test=False):


'''
Arguments:

This function takes in a training and testing directory, 
a tuple indicating how to resize the image, the color scale, 
the number of images to pull from the directory, 
and a boolean for process_test, which tells the function whether or not to create
a test generator to pull images from the testing directory.

Functionality:

This function creates image generators with the flow_from_directory method called off 
of them to pull images from their respective directories.
'''
    
    arg_dict = {'target_size':ts_tuple, 
                'color_mode':color, 
                'batch_size':batch_size}
    
    
    
    
    
    if process_test:
        
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='training', class_mode='binary')
        
        val_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='validation', class_mode='binary', shuffle=False)
        
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, **arg_dict, class_mode='binary', shuffle=False)

        
        return train_generator, val_generator, test_generator
    
    
    else:
    
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
       
        train_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='training', class_mode='binary', ) 

        val_generator = train_datagen.flow_from_directory(train_dir, **arg_dict, subset='validation', class_mode='binary', shuffle=False)
    
    
    
    
        return train_generator, val_generator
    


def visualize_cnn_test(history, model, train_generator, val_generator, test_generator):
   

'''
Arguments:

This function takes in model history, the model itself, the train generator,
the validation generator, and the test generator.

Functionality:

This function calculates accuracy, validation accuracy, loss, validation loss, 
recall, and validation recall. It then plots these metrics, 
evaluates the model on the training data, evaluates the model on the validation data, 
evaluates the model on the test data,
before finally plotting a confusion matrix showing the results 
of using the model to predict the testing data.

''' 
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    recall = history.history['recall']
    val_recall = history.history['val_recall']
 

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Test accuracy')
    plt.title('Training, and Test accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Test loss')
    plt.legend()
    plt.figure()
    plt.plot(epochs, recall, 'bo', label='Training recall')
    plt.plot(epochs, val_recall, 'b', label='Validation recall')
    plt.title('Training and Test recall')
    plt.legend()
    plt.show()
    
    
    print('')
    print('Training Evaluation:')
    model.evaluate(train_generator)
    print('')
    print('Validation Evaluation:')
    model.evaluate(val_generator)
    print('')
    print('Testing Evaluation:')
    model.evaluate(test_generator)
                                   
    
    
    preds = (model.predict(test_generator) > 0.5).astype('int32')                
                
    
    cm = confusion_matrix(test_generator.classes, preds)
    

    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    print('')
    print('Test Confusion Matrix')
    print('')
    
    cmd.plot();



def visualize_cnn(history, model, train_generator, val_generator):
    

'''
Arguments:

This function takes in model history, the model itself, the train generator and
the validation generator.

Functionality:

This function calculates accuracy, validation accuracy, loss, validation loss, 
recall, and validation recall. It then plots these metrics, 
evaluates the model on the training data, evaluates the model on the validation data, 
before finally plotting a confusion matrix showing the results 
of using the model to predict the validation data.

''' 
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    recall = history.history['recall']
    val_recall = history.history['val_recall']

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
    

    print('')
    print('Training Evaluation:')
    model.evaluate(train_generator)
    print('')
    print('Validation Evaluation:')
    model.evaluate(val_generator)
                                   
    
    
    
    preds = (model.predict(val_generator) > 0.5).astype('int32') 

    
    cm = confusion_matrix(val_generator.classes, preds)
    
      
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Pneumonia'])
    
    print('')
    print('Validation Confusion Matrix')
    print('')
    
    cmd.plot()
    ;


def cnn_model(cnn_filters,
              filters=[128],
              dense_filters=[512],
              kernel_size=(3,3),
              conv_activation='relu',
              input_shape=(256,256,1),
              pool_size=(2,2),
              five_by_five=False,
              five_activation='relu',
              l2_rate=0.01,
              conv_normal=False,
              conv_kernel_size=(5,5),
              conv_layer_activation='relu',
              dense_activation='relu',
              dense_reg=False,
              normal=False,
              output_nodes=1,
              output_activation='sigmoid',
              optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'Recall'],
              steps_per_epoch=60,
              epochs=20,
              validation_steps=20,
              process_test=False,
              train_directory=train_dir,
              test_directory=test_dir,
              ts_tuple=(256,256), 
              color='grayscale',
              batch_size=50,
              visualize=True):
    
    
    if not process_test:
        train_generator, val_generator = cnn_preprocessing(batch_size=batch_size, process_test=process_test)
    
    
    else:
        train_generator, val_generator, test_generator = cnn_preprocessing(batch_size=batch_size, process_test=process_test)
    
    
    print('ok')
    
    
    cnn_model = models.Sequential()
    
    
    for i, val in enumerate(cnn_filters):
    
        cnn_model.add(layers.Conv2D(cnn_filters[i], kernel_size=kernel_size, activation=conv_activation,
                                input_shape=input_shape))
    
        cnn_model.add(MaxPooling2D(pool_size))
        
    
    if five_by_five:
    
        cnn_model.add(Conv2D(64, (5, 5), activation=five_activation, kernel_regularizer=l2(l2=l2_rate)))
    
    
    for i, val in enumerate(filters):
        
        if conv_normal:

            
            cnn_model.add(layers.MaxPooling2D(pool_size))
        
            cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, use_bias=False))
            
            cnn_model.add(layers.BatchNormalization())
            
            cnn_model.add(layers.Activation(conv_layer_activation))
            
            
        else:
            
            cnn_model.add(layers.MaxPooling2D(pool_size))
        
            cnn_model.add(layers.Conv2D(filters[i], kernel_size=conv_kernel_size, activation=conv_layer_activation))

    
    
    cnn_model.add(MaxPooling2D(pool_size))
    
    
    cnn_model.add(layers.Flatten())
    
    
    
    for ind, value in enumerate(dense_filters):
    
        
        if normal:
            
            cnn_model.add(layers.Dense(dense_filters[ind], use_bias=False, kernel_regularizer=l2(l2=l2_rate)))
            
            cnn_model.add(layers.BatchNormalization())
            
            cnn_model.add(Activation(dense_activation))
        
        
        else:
            
            if dense_reg:
            
                cnn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation, kernel_regularizer=l2(l2=l2_rate)))
            
            else:
        
                cnn_model.add(layers.Dense(dense_filters[ind], activation=dense_activation))
            
    
    cnn_model.add(layers.Dense(output_nodes, activation=output_activation))

    
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=metrics)
    
    print('ok')
    
    cnn_class_weights = get_class_weights()
    
    hist = cnn_model.fit(train_generator, 
                         steps_per_epoch=steps_per_epoch, 
                         epochs=epochs,  
                         validation_data=(val_generator),
                         validation_steps=validation_steps,
                         class_weight=cnn_class_weights)
    
   
    if visualize:

        
        if process_test:
        
            visualize_cnn_test(hist, cnn_model, train_generator, val_generator, test_generator)
            
         
        else:
        
            visualize_cnn(hist, cnn_model, train_generator, val_generator)
            
    else:
        
        pass

    
    return cnn_model, hist  
