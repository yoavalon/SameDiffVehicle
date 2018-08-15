import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as matCol
import numpy as np
from datetime import datetime
import urllib
import cv2

a= tf.test.gpu_device_name()
print(a)

#Hyper-parameters
batchSize = 128
epochs = 50
learningRate = 0.001
cars = 100
preload = True


class Img : 

    def __init__(self, image, filename) : 
        self.image = np.asarray( image, dtype="int32" )
        self.filename = filename
        self.resolveLabel(self.filename)

    def resolveLabel(self, filename) :
        self.label = filename[:4]
        self.imgType = filename[4:7]

class SiameseNet:
    
    def __init__(self):
              
        self.left = tf.placeholder(tf.float32, [None, 64, 64, 1], name='left')
        self.right = tf.placeholder(tf.float32, [None, 64, 64, 1], name='right')

        #self.left = tf.placeholder(tf.float32, [None, 64], name='left')
        #self.right = tf.placeholder(tf.float32, [None, 64], name='right')
        
        self.label = tf.placeholder(tf.int32, [None,], name='label')                   
        self.loss = self.contrastive_loss()  

    def network(self, input, reuse = tf.AUTO_REUSE) :
        
        if (reuse):
          tf.get_variable_scope().reuse_variables()         
        
        with tf.name_scope("network") :          
          
          #input = tf.reshape(input, shape=[None, 64, 64, 1])
          
          with tf.variable_scope("conv1") as scope:            
            net = tf.contrib.layers.conv2d(input, 32, [7, 7],biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)            
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv1', reuse= reuse)
            
          with tf.variable_scope("conv2") as scope:
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv2', reuse= reuse)

          with tf.variable_scope("conv3") as scope:
            net = tf.contrib.layers.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv3', reuse= reuse)

          with tf.variable_scope("conv4") as scope:
            net = tf.contrib.layers.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, padding='SAME',
		        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv4', reuse= reuse)

          with tf.variable_scope("conv5") as scope:
            net = tf.contrib.layers.conv2d(net, 2, [2, 2], activation_fn=None, padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),scope=scope,reuse=reuse)
            net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')            
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True,  scope='conv5', reuse= reuse)                    
          
        net = tf.contrib.layers.flatten(net)   
                
        return net        
        
    def contrastive_loss(self):
      
      with tf.variable_scope("Siamese") as scope:
        model1= self.network(self.left)
        model2= self.network(self.right)
        self.label = tf.to_float(self.label)     
        y = self.label
      
      with tf.variable_scope("contrastive-loss") as scope:
        d = tf.sqrt(tf.reduce_sum(tf.pow(model1-model2, 2), 1, keep_dims=True))
        tmp= y * tf.square(d)    
        tmp2 = (1 - y) * tf.square(tf.maximum((margin - d),0))
      return tf.reduce_mean(tmp + tmp2) /2
     
def urlToImage(url):	
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)   #cv2.IMREAD_COLOR
 
	return image

urls1 = []
urls2 = []

def loadUrls() :
    
    for i in range(0,cars) :      
      urls1.append("https://raw.githubusercontent.com/yoavalon/Cars/master/" + str(i) + "-1.bmp")
      urls2.append("https://raw.githubusercontent.com/yoavalon/Cars/master/" + str(i) + "-2.bmp")            
    
    return np.array(urls1), np.array(urls2)

urls1, urls2 = loadUrls()
  
def getImage(num, index) :  

  if index == 1 :
    url = 'https://raw.githubusercontent.com/yoavalon/Cars/master/' + urls1[num]
  if index == 2:
    url = 'https://raw.githubusercontent.com/yoavalon/Cars/master/' + urls2[num]
  img = urlToImage(url)  
  img = np.resize(img, (64,64,1))
  
  return img  

images1 = []
images2 = []

def PreloadImages() : 
    for i in range(0,cars) :      
      images1.append(getImage(i,1))
      images2.append(getImage(i,2))  

if(preload) :
  PreloadImages()
  print("images pre loaded")

g = tf.Graph() 
sess = tf.InteractiveSession(graph=g)

model = SiameseNet();
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(model.loss)

tf.initialize_all_variables().run()
lossList = []

for step in range(epochs):    
  
  leftIndices = np.random.randint(cars, size=batchSize)
  otherIndices = np.random.randint(cars, size=batchSize)
  same = np.random.randint(2, size=batchSize)      
  rightIndices = np.where(same == 1, leftIndices, otherIndices)
    
  if(preload) :
    leftImages = [images1[item] for item in leftIndices]
    rightgImages = [images2[item] for item in rightIndices]
  else:     
    leftImages = [getImage(item,1) for item in leftIndices]
    rightImages = [getImage(item,2) for item in rightIndices]
    
  _, loss_v = sess.run([optimizer, model.loss], feed_dict={ model.left: leftImages, model.right: rightImages , model.label: same})
  lossList.append(loss_v)    
    
  print('step %3d:  loss: %.6f ' % (step, loss_v))       
      
# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
