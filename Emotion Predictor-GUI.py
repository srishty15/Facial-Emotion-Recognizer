from tkinter import *
import time
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

#Main Frame
root=Tk()
root.title("FaceMoji :)")
root.configure(background='#C8E0EC')

#Upload Image
def upload_image():
    root1 = Tk()
    root1.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[(("Picture File", "*.jpg;*.png"))])
    img = Image.open(file_path)
    file_names = str(int(time.time()))
    print(file_names)
    img.save("C:/Users/9911v/PycharmProjects/Facial Emotion Detector/abc.png")
    tkimage = ImageTk.PhotoImage(img)
    root1.mainloop()

#Predict the Answer
def predict_image():
        # get the data
        filname = 'fer2013.csv'
        label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        def getData(filname):
            Y = []
            X = []
            first = True
            for line in open(filname):
                if first:
                    first = False
                else:
                    row = line.split(',')
                    Y.append(int(row[0]))
                    X.append([int(p) for p in row[1].split()])

            X, Y = np.array(X) / 255.0, np.array(Y)
            return X, Y

        X, Y = getData(filname)
        num_class = len(set(Y))

        # To see number of training data point available for each label
        def balance_class(Y):
            num_class = set(Y)
            count_class = {}
            for i in range(len(num_class)):
                count_class[i] = sum([1 for y in Y if y == i])
            return count_class

        balance = balance_class(Y)

        # keras with tensorflow backend
        N, D = X.shape
        X = X.reshape(N, 48, 48, 1)

        # Split in  training set : validation set :  testing set in 80:10:10
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
        y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
        y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

        batch_size = 128
        epochs = 124

        # Main CNN model with four Convolution layer & two fully connected layer
        def baseline_model():
            # Initialising the CNN
            model = Sequential()

            # 1 - Convolution
            model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=(48, 48, 1)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # 2nd Convolution layer
            model.add(Conv2D(128, (5, 5), border_mode='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # 3rd Convolution layer
            model.add(Conv2D(512, (3, 3), border_mode='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # 4th Convolution layer
            model.add(Conv2D(512, (3, 3), border_mode='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # Flattening
            model.add(Flatten())

            # Fully connected layer 1st layer
            model.add(Dense(256))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

            # Fully connected layer 2nd layer
            model.add(Dense(512))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))

            model.add(Dense(num_class, activation='sigmoid'))

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
            return model

        def baseline_model_saved():
            # load json and create model
            json_file = open('model_4layer_2_2_pool.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights from h5 file
            model.load_weights("model_4layer_2_2_pool.h5")
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
            return model

        is_model_saved = True

        # If model is not saved train the CNN model otherwise just load the weights
        if (is_model_saved == False):
            # Train model
            model = baseline_model()
            # Note : 3259 samples is used as validation data &   28,709  as training samples

            model.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      validation_split=0.1111)
            model_json = model.to_json()
            with open("model_4layer_2_2_pool.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model_4layer_2_2_pool.h5")
            print("Saved model to disk")
        else:
            # Load the trained model
            print("Load model from disk")
            model = baseline_model_saved()

        # Model will predict the probability values for 7 labels for a test image
        score = model.predict(X_test)
        print(model.summary())

        new_X = [np.argmax(item) for item in score]
        y_test2 = [np.argmax(item) for item in y_test]

        # Calculating categorical accuracy taking label having highest probability
        accuracy = [(x == y) for x, y in zip(new_X, y_test2)]
        print(" Accuracy on Test set : ", np.mean(accuracy))

        def emotion_analysis(emotions):
            objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            y_pos = np.arange(len(objects))

            plt.bar(y_pos, emotions, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('percentage')
            plt.title('emotion')

            plt.show()

        # make prediction for custom image out of test set

        img = image.load_img("abc.png", grayscale=True, target_size=(48, 48))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x /= 255

        custom = model.predict(x)
        emotion_analysis(custom[0])

        x = np.array(x, 'float32')
        x = x.reshape([48, 48])

        plt.gray()
        plt.imshow(x)
        plt.show()


#Adding the logo and heading
imagelogo = PhotoImage(file='logo.png')
logo1 = Label(root, image=imagelogo,bg='#C8E0EC',height = 270,width=400)
logo1.place(x=75, y=-20)
lable1 = Label(root, text="FaceMoji",background='#C8E0EC',fg="#FF8699",font = "Arial 100 bold").place(x=500, y=30)
imagelogo1 = PhotoImage(file='logo.png')
logo2 = Label(root, image=imagelogo1,bg='#C8E0EC',height = 270,width=400)
logo2.place(x=1090, y=-20)

#Adding Text
label1 = Label(root, text="Hey! \n Welcome to FaceMoji :) \n It is a facial emotion recognition technology \n capable of identifying or verifying \n a person’s emotion from a digital image.",background='#C8E0EC',fg="black",font = "Arial 15 bold", borderwidth=2,relief="solid",highlightcolor="#D2691E")
label1.place(x=550, y=300)

#Upload Image Button
button1 = Button(root, text="Upload the Image",command=upload_image,width=15,background='#FF8699',font = "Arial 15",fg="white")
button1.place(x=400, y=550)

#Predict Button
button2 = Button(root, text="Predict",command=predict_image,width=15,background='#FF8699',font = "Arial 15",fg="white")
button2.place(x=900, y=545)

root.mainloop()