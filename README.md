# Feed Forward Neural Network

Implemented stochastic gradient descent for the multi-layer neural network shown above. The algorithm works for any number of hidden layers.

![image](https://user-images.githubusercontent.com/64325043/226217208-9b87a170-3440-4d0d-a324-5fd6eeb2d65e.png)
![image](https://user-images.githubusercontent.com/64325043/226217192-ada3b6ee-6cae-406d-8cd7-9666b81063a9.png)

The (unregularized) cross-entropy cost function is :

![image](https://user-images.githubusercontent.com/64325043/226217270-8c7c27cf-5f45-4eea-ae86-a50cfe5fcae1.png)




Gradient function is checked for correctness using the check_grad method.

After Training;
The best parameters found for the set are:
Epochs : 150
Learning Rate : 0.001
Mini Batch Size: 128

The accuracy obtained was # 87.6%

![image](https://user-images.githubusercontent.com/64325043/226217358-9c9f6a00-f18b-4ab3-9be9-59422c1d3d8a.png)


The first layer of weights W that are learned after training the best neural network were visualized. 

![image](https://user-images.githubusercontent.com/64325043/226217411-92dcc796-1ee6-4d73-81c2-dd01b67d5551.png)


