package layers

//

import (
	// "fmt"
	// "math"
	// // "math/big"
	"github.com/luckylwk/godeep/go-deep/utils"
)

//

type BaseLayer interface {
    Name() string
    // 
    Init( int, []utils.ActivationFunction )
    InitParameters() // Function to generate the parameters inside the layer (weights, biases, kernels).
    //
    FeedForward( []float64 )
    CalculateWeightsSum( float64 ) float64
    CalculateGradients()
    //
 	UpdateWeights( float64, float64, float64, float64 )
    CalculateDeltaRegular([]float64)
    CalculateDeltaOutputSquaredError([]float64)
    CalculateDeltaOutputCrossEntropy([]float64)
    // Getters and setters for dimensions.
    GetInputDim() int
    SetInputDim(int)
    GetOutputDim() int
    SetOutputDim(int)
    GetActivations() []float64
    SetActivations([]float64)
    GetActivationFunction() string
    SetActivationFunction([]utils.ActivationFunction)
    GetWeights() [][]float64
    GetDeltas() []float64
    
    // SetLearningRate(*float64)
}
