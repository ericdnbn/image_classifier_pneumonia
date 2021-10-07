import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight


def get_images(filepath):
    '''
    Given a filepath for a folder of images, return a list of those images
    as arrays of dtype=uint8.
    '''
    # List of filenames
    filenames = os.listdir(filepath)
    # List of full filepaths to each image
    filepaths = [os.path.join(filepath, name) for name in filenames]
    # Return list of files as raw image arrays
    return [mpimg.imread(img) for img in filepaths]


def preprocess_dense_data(train_dir, 
                          test_dir=None, 
                          target_size=(256, 256),
                          batch_size=100,
                          color_mode='grayscale',
                          class_mode='binary'):
    '''
    Prepare image data for fitting in a fully connected dense neural network by
    getting data from directories, scaling, and reshaping.
    
    Parameters: directories for training data (optional: testing data.
    
    Returns: X_train, X_val, X_test, train_labels, val_labels, test_labels
    '''
    # Param-dict for 'flow_from_directory'
    directory_kwargs = {'target_size':target_size,
                        'batch_size':batch_size,
                        'color_mode':color_mode,
                        'class_mode':class_mode}
    
    # Get data from directories and scale        
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(train_dir, subset='training', **directory_kwargs)
    val_generator = train_datagen.flow_from_directory(train_dir, subset='validation', **directory_kwargs)
    
    # Create the datasets
    train_images, train_labels = next(train_generator)
    val_images, val_labels = next(val_generator)
    
    # Create testing dataset if test_dir is provided
    if test_dir:
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_directory, **directory_kwargs)
        test_images, test_labels = next(test_generator)
    
    # Prepare images for modeling by reshaping
    def reshape_images(images):
        return images.reshape(images.shape[0], -1)
    
    X_train = reshape_images(train_images)
    X_val = reshape_images(val_images)

    if test_dir:
        X_test = reshape_images(test_images)
        return X_train, X_val, X_test, train_labels, val_labels, test_labels
    else:
        return X_train, X_val, train_labels, val_labels

    
def preprocess_cnn_data(train_dir, 
                        test_dir=None, 
                        target_size=(256, 256),
                        batch_size=100,
                        color_mode='grayscale',
                        class_mode='binary'):
    '''
    Prepare image data for fitting in a convolutional neural network by
    getting data from directories and scaling.
    
    Parameters: directories for training data (optional: testing data)
    
    Returns: train_generator, val_generator (optional: test_generator)
    '''
    # Param-dict for 'flow_from_directory'
    directory_kwargs = {'target_size':target_size,
                        'batch_size':batch_size,
                        'color_mode':color_mode,
                        'class_mode':class_mode}
    
    # Make generators        
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_directory(train_dir, subset='training', **directory_kwargs)
    val_generator = train_datagen.flow_from_directory(train_dir, subset='validation', shuffle=False, **directory_kwargs)
    
    # Create testing generator if test_dir is provided
    if test_dir:
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_directory, **directory_kwargs)

    if test_dir:
        return train_generator, val_generator, test_generator
    else:
        return train_generator, val_generator
    

def visualize_training_results(results, model=None, X_train=None, train_labels=None, X_val=None, val_labels=None):
    '''
    Plot the training and validation data from a trained NN model, given the results/history.
    Plot accuracy, recall, and loss.
    
    If model and data are provided, print evaluation of training and validation data
    and plot confusion matricies.
    '''
    # Training history
    history = results.history
    
    # Accuracy
    plt.figure()
    plt.plot(history['val_acc'])
    plt.plot(history['acc'])
    plt.legend(['Validation acc', 'Training acc'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    # Recall
    plt.figure()
    plt.plot(history['val_recall'])
    plt.plot(history['recall'])
    plt.legend(['Validation recall', 'Training recall'])
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.show()
    
    # Loss
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['Validation loss', 'Training loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # Evaluation
    if model and (X_train is not None) and (train_labels is not None):
        print('Training eval:')
        results_train = model.evaluate(X_train, train_labels)
    if (X_val is not None) and (val_labels is not None):
        print('\nValidation eval:')
        results_val = model.evaluate(X_val, val_labels)
        
    # Confusion matrix
    if model and (X_train is not None) and (train_labels is not None):
        y_preds = (model.predict(X_train) > 0.5).astype('int32')
        ConfusionMatrixDisplay(confusion_matrix(train_labels, y_preds), 
                               display_labels=['Train_Normal', 'Train_Pneumonia']).plot()
    
    if model and (X_val is not None) and (val_labels is not None):
        y_val_preds = (model.predict(X_val) > 0.5).astype('int32')
        ConfusionMatrixDisplay(confusion_matrix(val_labels, y_val_preds),
                               display_labels=['Val_Normal', 'Val_Pneumonia']).plot();
        
    return results

def visualize_results_cnn(results, model=None, train_gen=None, val_gen=None):
    '''
    Plot the training and validation data from a trained CNN model, given the results/history.
    Plot accuracy, recall, and loss.
    
    If model and generators are provided, print evaluation of training and validation data
    and plot confusion matricies.
    '''
    # Training history
    history = results.history
    
    # Accuracy
    plt.figure()
    plt.plot(history['val_acc'])
    plt.plot(history['acc'])
    plt.legend(['Validation acc', 'Training acc'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    # Recall
    plt.figure()
    plt.plot(history['val_recall'])
    plt.plot(history['recall'])
    plt.legend(['Validation recall', 'Training recall'])
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.show()
    
    # Loss
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['Validation loss', 'Training loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    # Evaluations
    if model and train_gen:
        print('Training eval:')
        results_train = model.evaluate(train_gen)
    if model and val_gen:
        print('\nValidation eval:')
        results_val = model.evaluate(val_gen)
        
    # Confusion matricies
      # Training cm not produced because labels do not match order of samples with .predict()
#     if model and train_gen:
#         y_preds = (model.predict(train_gen) > 0.5).astype('int32')
#         ConfusionMatrixDisplay(confusion_matrix(train_gen.labels, y_preds), 
#                                display_labels=['Train_Normal', 'Train_Pneumonia']).plot();
    
    if model and val_gen:
        y_val_preds = (model.predict(val_gen) > 0.5).astype('int32')
        ConfusionMatrixDisplay(confusion_matrix(val_gen.labels, y_val_preds),
                               display_labels=['Val_Normal', 'Val_Pneumonia']).plot();
        
    return results


def get_class_weights(train_labels):
    '''
    Calculate class weights from training label array (`train_labels`).
    Return dictionary: {class:weight}
    '''
    # From https://stackoverflow.com/questions/42586475/is-it-possible-to-automatically-infer-the-class-weight-from-flow-from-directory
    # Calculate floats/raw class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(train_labels),
                                                      y=train_labels)

    # Convert to dict for .fit(class_weight= ) param
    nn_class_weights = {0:class_weights[0],
                        1:class_weights[1]}
    
    return nn_class_weights


def show_misclassfied_images(model, X, labels, image_shape=(256, -1), classes=None, verbose=False, size=(20, 15)):
    '''
    Given a model, X set of images, and ground truth labels, plot the images
    that the model misclassifies, separating false positives and false negatives.
    
    Parameters: 
    model: trained model
    X: array of flattened images
    labels: array of ground truth labels
    image_shape: tuple of original image shape
    classes: dictionary of binary class labels
    verbose: return array of indicies of misclassfied images
    size: size of resulting image plot arrays
    
    Returns:
    None or array of indicies of misclassfied images 
    '''
    # Get indicies of misclassfied images
    # From https://stackoverflow.com/questions/39300880/how-to-find-wrong-prediction-cases-in-test-set-cnns-using-keras
    incorrects = np.nonzero((model.predict(X)>0.5).reshape((-1,)) != labels)[0]
                             
    # Dictionary for labels
    if classes is None:
        classes = {0:'Normal',
                   1:'Pneumonia'}
    
    # Lists of indicies
    false_positives = [i for i in incorrects if labels[i]==0]
    false_negatives = [i for i in incorrects if labels[i]==1]

    # False positives
    if len(false_positives)>0:
        rows = (len(false_positives)//5)+1
        cols = 5
        fig, axs = plt.subplots(rows, cols, figsize=(rows*4, cols*3))
        fig.suptitle('False Positives', fontsize=15, y=0.9)
        for i, ax in enumerate(axs.ravel()):
            try:
                ax.imshow(X[false_positives[i]].reshape(256, -1), cmap='gray')
            except:
                break

    # False negatives
    if len(false_negatives)>0:
        rows = (len(false_negatives)//5)+1
        cols = 5
        fig, axs = plt.subplots(rows, cols, figsize=(rows*4, cols*3))
        fig.suptitle('False Negatives', fontsize=15, y=0.9)
        for i, ax in enumerate(axs.ravel()):
            try:
                ax.imshow(X[false_negatives[i]].reshape(256, -1), cmap='gray')
            except:
                break
   
    if verbose:
        return incorrects
    else:
        return
