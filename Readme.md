# Deep learning?



## A first look at Neural Network

```
from tensorflow import keras
from tensorflow.keras import layers 
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

block ตัวนี้เรียกว่า  1 layer 
* ชั้นแรก มี 512 (Dense) และใช้ฟังก์ชัน activation เป็น "relu" ซึ่งใช้ในการเปลี่ยนแปลงข้อมูลให้มีความเป็นประโยชน์มากขึ้น
* ชั้นสอง 10 (Dense) และใช้ funtion softmax ใช้สำหรับจำแนกข้อมูลเป็น 10 ประเภท
* หน้าที่ของแต่ละชั้นคือการกรองข้อมูล

การทำให้ Model พร้อทสำหรับการฝึก
* An optimizer  เป็นกลไกในการ update ตัวเองโดยมี training data เป็นพื้นฐาน เพื่อเพิ่มประสิทธิภาพ
* A loss function การวัดประสิทธิภาพ model เพื่อดูว่าเรากำลังไปถูกทางหรือไม้
* Metrics to moniter during traning and testing  จะสนใจเฉพาะความแม่นยำ

```
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```

ก่อนการ train model จะมีการ reshaping data และ scaling data  ให้ข้อมูลอยู่ใน [0,1]  (60000,28,28) หมายถึงมีภาพ 60000 ภาพขนาดแต่ละภาพคือ 28 x 28 พิกเซล

การปรับขนาดของข้อมูล (Scaling): ค่าของพิกเซลในภาพเดิมมีค่าระหว่าง 0 ถึง 255 และเป็นชนิดข้อมูล uint8 การปรับขนาดนี้จะทำให้ค่าทั้งหมดอยู่ในช่วงระหว่าง 0 ถึง 1 และเปลี่ยนชนิดข้อมูลเป็น float32 เพื่อความแม่นยำในการคำนวณ

```
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
```

การเทรนใน Keras  จะเรียกว่า fit() method 
```
>>> model.fit(train_images, train_labels, epochs=5, batch_size=128)
Epoch 1/5
60000/60000 [===========================] - 5s - loss: 0.2524 - acc: 0.9273 Epoch 2/5
51328/60000 [=====================>.....] - ETA: 1s - loss: 0.1035 - acc: 0.9692
```
ดูค่าความแม่นยำถ้า train ครบ **98.8 %** 

```
>>> test_digits = test_images[0:10]
>>> predictions = model.predict(test_digits)
>>> predictions[0]
array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01, 2.6657624e-08, 3.8127661e-07], dtype=float32)
```
แต่ละค่าของ index i จะเกี่ยวข้องกับค่าความน่าจะเป็นที่จะเป็นเลขนั้นๆ  ดังนั้นค่าตัวเลขที่สูงสุดคือ index ที่ 7 0.99999106 เกือบ 1 ตามโมเดลจะต้องเป็น 7 

```
 >>> predictions[0].argmax()
        7
        >>> predictions[0][7]
        0.99999106

```


```
test_loss, test_acc = model.evaluate(test_images, 
test_labels) >>> print(f"test_acc: {test_acc}")
test_acc: 0.9785
```
จากผลความแม่นยำจาก test set   97.85 % ซึ่งน้อยกว่า training set  แบบนี้แสดงว่า overfitting 

# deep_learning_sum_book
