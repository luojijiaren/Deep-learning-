import tensorflow as tf
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

#load data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])


W1=tf.Variable(tf.random_uniform([784,50],minval=-tf.sqrt(6.0)/(tf.sqrt(784.0+50.0)),maxval=tf.sqrt(6.0)/(tf.sqrt(784+50.0)),seed=1))
W2=tf.Variable(tf.random_uniform([50,10],minval=-tf.sqrt(6.0)/(tf.sqrt(10+50.0)),maxval=tf.sqrt(6.0)/(tf.sqrt(10+50.0)),seed=3))
b1=tf.Variable(tf.ones([50]))
b2=tf.Variable(tf.ones([10]))

#construct 3 layer network
h1=tf.nn.relu(tf.matmul(x,W1)+b1)
y_=tf.nn.softmax(tf.matmul(h1,W2)+b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#GSD with mini-batch
for epoch in range(50):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/100)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(100)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_step,cross_entropy], feed_dict={x: batch_x, y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if epoch >=30:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
print("Optimization Finished!")

#get accuracy and cost of 
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
a_test=sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
c_test=sess.run(cross_entropy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
print("The accuracy on the testing set: "+"{0: .2%}".format(a_test))
print("The cross-entropy cost on the testing set: {}".format('%.3f' %c_test))