package main

//

import (
	"fmt"
	"github.com/luckylwk/godeep/go-deep"
	"github.com/luckylwk/godeep/go-deep/layers"
    // "github.com/luckylwk/go-neural/godeep/utils"
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