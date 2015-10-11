package layers

//

import (
	"fmt"
	"math"
	// "math/big"
	"github.com/luckylwk/godeep/go-deep/utils"
)

//

type DenseLayer struct {
	InputDim int
	OutputDim int
	Activation utils.ActivationFunction
	//
	Weights [][]float64
	Biases []float64
	//
	Input []float64
	Zetas []float64
	Activations []float64
	// Delta's and Gradients
	Deltas []float64
	GradientsWeights [][]float64
	GradientsBiases []float64
	// Model Parameters
	LearningRate *float64
}

func (f *DenseLayer) Name() string {
    return "Dense Layer (Fully Connected)"
}

func (self *DenseLayer) Init( layerSize int, activation []utils.ActivationFunction ) {
	// Set the size
	self.SetOutputDim(layerSize)
	// Set the layer activation
	self.SetActivationFunction(activation)
	fmt.Println("\tLayer Activation function: ", self.GetActivationFunction())
	// Set the layer regularisation

	// Set the layer dropout

}

func (self *DenseLayer) InitParameters() {
	self.Weights = utils.WeightsUniformGlorot(self.InputDim,self.OutputDim)
	self.Biases = utils.VectorConstant( self.OutputDim, 0.0 )
	// Initialise the output calculations all to zero.
	self.Zetas = utils.VectorConstant( self.OutputDim, 0.0 )
	self.Activations = utils.VectorConstant( self.OutputDim, 0.0 )
	self.Deltas = utils.VectorConstant( self.OutputDim, 0.0 )
	self.GradientsWeights = utils.MatrixConstant( self.InputDim, self.OutputDim, 0.0 )
	self.GradientsBiases = utils.VectorConstant( self.OutputDim, 0.0 )
}

func (self *DenseLayer) FeedForward( inputVector []float64 ) {
	// Matrix wise we are multiplying an 1-by-f (features) vector
	// with a f-by-Lu weight matrix (features by number of units in layer L)
	// The output of this function should be an 1-by-Lu vector with unit-activations.
	// So we are going to run the outer loop on the outer dimension of the weight matrix
	self.Input = inputVector
	for o := 0; o < self.OutputDim; o++ {
		unitZeta := 0.0
		unitZeta += 1.0 * self.Biases[o]
		for i := 0; i < self.InputDim; i++ {
			// Now. For each feature we multiply with the weight and keep summing up.
			unitZeta += inputVector[i] * self.Weights[i][o]
		}
		self.Zetas[o] = unitZeta
		// Now we have the zeta, we calculate the activation using the sigmoid.
		// We need to store the zeta for the calculation of gradients at the later stage.
		self.Activations[o] = self.Activation.Fn(unitZeta)
	}
}

func (self *DenseLayer) CalculateGradients() {
	for o := 0; o < self.OutputDim; o++ {
		self.GradientsBiases[o] += self.Deltas[o]
	}
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			self.GradientsWeights[i][o] += self.Input[i] * self.Deltas[o]
		}
	}
}

func (self *DenseLayer) UpdateWeights( m float64, learningRate float64, l2 float64, l1 float64 ) {
	// Make sure to set the gradients to zero.
	for o := 0; o < self.OutputDim; o++ {
		self.Biases[o] = self.Biases[o] - (learningRate/m) * self.GradientsBiases[o]
		self.GradientsBiases[o] = 0.0
	}
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			l1pen := 0.0 // learningRate * l1 * float64(big.Sign(self.Weights[i][o]))
			l2pen := learningRate * l2 * self.Weights[i][o]
			self.Weights[i][o] = self.Weights[i][o] - (learningRate/m) * self.GradientsWeights[i][o] - l1pen - l2pen
			// self.Weights[i][o] = (1-learningRate*l2) * self.Weights[i][o] - (learningRate/m) * self.GradientsWeights[i][o]
			self.GradientsWeights[i][o] = 0.0
		}
	}
}

func (self *DenseLayer) CalculateWeightsSum( power float64 ) float64 {
	output := 0.0
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			output += math.Pow(math.Abs(self.Weights[i][o]),power)
		}
	}
	return output
}





// This needs to be optimised.

func (self *DenseLayer) CalculateDeltaRegular( Wd []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		// Hademard-product. Pointwise multiplication with the differential.
		self.Deltas[o] = Wd[o] * self.Activation.FnPrime(self.Zetas[o])
	}
}

func (self *DenseLayer) CalculateDeltaOutputSquaredError( y []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		// Hademard-product. Pointwise multiplication with the differential.
		self.Deltas[o] = (self.Activations[o] - y[o]) * self.Activation.FnPrime(self.Zetas[o])
	}
}

func (self *DenseLayer) CalculateDeltaOutputCrossEntropy( y []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		self.Deltas[o] = self.Activations[o] - y[o]
	}
}


// Getters and Setters (needed for Interface)

func (self *DenseLayer) GetInputDim() int {
	return self.InputDim
}
func (self *DenseLayer) SetInputDim(dim int) {
	self.InputDim = dim
}
func (self *DenseLayer) GetOutputDim() int {
	return self.OutputDim
}
func (self *DenseLayer) SetOutputDim(dim int) {
	self.OutputDim = dim
}
func (self *DenseLayer) GetActivations() []float64 {
	return self.Activations
}
func (self *DenseLayer) SetActivations(act []float64) {
	self.Activations = act
}
func (self *DenseLayer) GetActivationFunction() string {
	return self.Activation.Name()
}
func (self *DenseLayer) SetActivationFunction(act []utils.ActivationFunction) {
	self.Activation = act[0]
}
func (self *DenseLayer) GetWeights() [][]float64 {
	return self.Weights
}
func (self *DenseLayer) GetDeltas() []float64 {
	return self.Deltas
}


// Getters and Setters for MODEL parameters.

// func (self *DenseLayer) SetLearningRate(lr *float64) {
// 	self.LearningRate = lr
// }
