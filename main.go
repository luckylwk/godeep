package main

//

import (
	"fmt"
	"github.com/luckylwk/godeep/go-deep"
	"github.com/luckylwk/godeep/go-deep/layers"
    "github.com/luckylwk/godeep/go-deep/utils"
)

//

func main() {
	
	trainingDataFile := "data/train-images-idx3-ubyte"
	trainingLabelsFile := "data/train-labels-idx1-ubyte"
    testingDataFile := "data/t10k-images-idx3-ubyte"
    testingLabelsFile := "data/t10k-labels-idx1-ubyte"

    fmt.Println("\n\n *** LOADING TRAINING DATA")
	var labelData []byte
    var imageData [][]byte
    imageData, _, _ = DataLoadImages(OpenFile(trainingDataFile))
	labelData = DataLoadLabels(OpenFile(trainingLabelsFile))
    trainingInputs := PreprocessGlobalContrastStandardisation(imageData, 255.0)
    trainingTargets := PreprocessLabelsToVector(labelData, 10)

    // Print samples of the input-data to make sure everything is correct.
    fmt.Println("\tTraining labels: ", trainingTargets[0], " | resembling a ", labelData[0])
    imageData = nil
    labelData = nil




    fmt.Println("\n\n *** BUILDING MODEL")
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



    // TEST OUT OF SAMPLE.

    var testImageData [][]byte
    var testLabelData []byte
    fmt.Println("\n\n *** LOADING TEST DATA")
    testImageData, _, _ = DataLoadImages(OpenFile(testingDataFile))
    testLabelData = DataLoadLabels(OpenFile(testingLabelsFile))
    testingInputs := PreprocessGlobalContrastStandardisation(testImageData,255.0)
    testingTargets := PreprocessLabelsToVector(testLabelData,10)
    testImageData = nil
    testLabelData = nil

    model.Predict( testingInputs, testingTargets )

}