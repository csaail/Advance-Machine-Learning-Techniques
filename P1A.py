#P1A:  Design a simple machine learning model to train the training instances and test the same using Python.
import random
from sklearn.linear_model import LinearRegression

print("Name: Saail Chavan 016")

feature_set = []
target_set = []

rows = 200
limit = 2000

for i in range(0,rows):
  x = random.randint(0,limit)
  y = random.randint(0,limit)
  z = random.randint(0,limit)
  g= 10*x + 2*y + 3*z
  
  print("x=",x,"\ty=",y,"\tz=",z,"\tg=",g);
  feature_set.append([x,y,z])
  target_set.append(g)

model = LinearRegression()
model.fit(feature_set,target_set)

Test_Data = [[1,2,1]]

prediction = model.predict(Test_Data)
prediction = model.predict(Test_Data)

print('Prediction:'+str(prediction)+'\t'+ 'Coefficient:'+str(model.coef_))
print("Name : Saail Chavan 016")
