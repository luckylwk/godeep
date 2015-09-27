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

// Create new layer and add to model.
layer1 := layers.BaseLayer{}
layer1.Init(40)
model.Add( &layer1 )

// Create new layer and add to model.
layer2 := layers.BaseLayer{}
layer2.Init(20)
model.Add( &layer2 )

// Create new layer and add to model.
layer3 := layers.BaseLayer{}
layer3.Init(10)
model.Add( &layer3 )

model.Compile()

model.Train(trainingInputs,trainingTargets,10,20)
~~~

Regularisation, Dropout and alternative activation functions (to sigmoidal) have not been implemented yet.