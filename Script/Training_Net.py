import keras
import argparse
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt

args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a Training NLP Net program')
    parser.add_argument("data", help="input training file in pickl format (*.pkl).")
    parser.add_argument("model", help="input model structure file in HDF5 format (*.h5).")
    parser.add_argument("-e", "--epoch", help="Number of training epoch.", default=30, type=int)
    parser.add_argument("-b","--batch_size", help="number of batch size", default = 512, type=int)
    parser.add_argument("-t","--train_val_split", help="persent of val(test) split", default = 0.3, type=float)
    #parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args

def show_model_effect(history):
    """
    Visualize the changes of evaluation 
    indicators in the training process
    like Tensorboard
    """

    # summarize history for accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(self.model_path+"/Performance_accuracy.jpg")

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(self.model_path+"/Performance_loss.jpg")


if __name__ == '__main__':
    args = ParseArgs()

    #load model
    model = keras.models.load_model(args.model)
    model.summary()
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    #load data
    with open(args.data, 'rb') as f:
        data_f = pickle.load(f)
    
    data_X = data_f['X']
    data_Y = data_f['Y']

    train_x, val_x, train_y, val_y = train_test_split(data_X,
                    data_Y, test_size=args.train_val_split, random_state=2019)
    # Training
    model_path = '../model'
    # Add Checkpoint
    model_info = "/cnn_bilstm_am_model_classNum2_"
    epoch_info = 'model-ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5'
    ckpt_fn = model_path + model_info + epoch_info
    ckpt = ModelCheckpoint(filepath=ckpt_fn, monitor='val_acc', save_best_only=False, mode='max')
    print(ckpt_fn)
    early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=1)
    callbacks = [ckpt, early_stopping]

    # Training the model

    history = model.fit(train_x, train_y, epochs=args.epoch, batch_size=args.batch_size,
                        validation_data=(val_x, val_y), callbacks=callbacks, verbose=1)

    show_model_effect(history)