from keras.models import Sequential
from keras.layers import Dense, Dropout


def get_ann():
    model = Sequential()
    model.add(Dense(units=78,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=39,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=19,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=4,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model



