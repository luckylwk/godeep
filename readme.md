# Deep Learning Framework in GOLANG

Build and run using:

~~~bash
$ go build .

$ ./godeep
~~~

Model setup looks like

~~~go
model := godeep.Sequential{}
model.Init(len(trainingInputs[0]),"CROSSENTROPY")

// Create new layers and add to model.
layer1 := layers.DenseLayer{}
layer1.Init(40,[]utils.ActivationFunction{&utils.Sigmoid{}})
layer2 := layers.DenseLayer{}
layer2.Init(20,[]utils.ActivationFunction{&utils.Tanh{}})
layer3 := layers.DenseLayer{}
layer3.Init(10,[]utils.ActivationFunction{&utils.Sigmoid{}})

// Add the created layers.
model.Add( []layers.BaseLayer{&layer1,&layer2,&layer3} )

// Compile the model.
model.Compile()

model.Train(trainingInputs,trainingTargets,2,20)
~~~

To implement next:

- Dropout Layers
- Recurrent Layers



// Notes:
https://github.com/luciotato/golang-notes/blob/master/OOP.md