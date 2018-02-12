"""

23.
Mycobacterium phage Shipwreck, complete genome
48,670 bp linear DNA
Accession: NC_031261.1 GI: 1070639280
Assembly BioProject Taxonomy
GenBankFASTAGraphics


"""

from keras.layers import Dense
from keras.models import Sequential


model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=128))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='binary_crossentropy',
              opimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)