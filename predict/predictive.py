import pickle
import numpy as np
load_model=pickle.load(open("D:\diabetiespredict\predict\model.sav","rb"))
scaler=pickle.load(open("D:\diabetiespredict\predict\scaler.sav","rb"))
input_data=(8	,183	,64	,0,	0,	23.3,	0.672	,32)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction=load_model.predict(std_data)
print(prediction[0])

if(prediction[0]==0):
    print("the person is not diabetic")
else:
    print("the person is diabetic")


