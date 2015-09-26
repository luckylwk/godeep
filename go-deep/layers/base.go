package layers

//

import (
	// "math/rand"
	"math"
	"github.com/luckylwk/godeep/go-deep/utils"
)

//

type BaseLayer struct {
	// Size.
	InputDim int
	OutputDim int
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
}

func (self *BaseLayer) Init( layerSize int ) {
	// Set the size
	self.OutputDim = layerSize
	// Set the layer activation

	// Set the layer regularisation

	// Set the layer dropout

}

func (self *BaseLayer) InitParameters() {
	self.Weights = utils.WeightsUniformGlorot(self.InputDim,self.OutputDim)
	self.Biases = make([]float64, self.OutputDim)
	// Initialise the output calculations all to zero.
	self.Zetas = make([]float64, self.OutputDim)
	self.Activations = make([]float64, self.OutputDim)
	self.Deltas = make([]float64, self.OutputDim)
	self.GradientsWeights = utils.MatrixConstant( self.InputDim, self.OutputDim, 0.0 )
	// self.GradientsBiases = make([]float64, self.OutputDim) // Inits to zeros 
	self.GradientsBiases = utils.VectorConstant( self.OutputDim, 0.0 )
}

func (self *BaseLayer) FeedForward( inputVector []float64 ) {
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
		self.Activations[o] = utils.Sigmoid(unitZeta)
	}
}

func (self *BaseLayer) CalculateGradients() {
	for o := 0; o < self.OutputDim; o++ {
		self.GradientsBiases[o] += self.Deltas[o]
	}
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			self.GradientsWeights[i][o] += self.Input[i] * self.Deltas[o]
		}
	}
}

func (self *BaseLayer) UpdateWeights( m float64, learningRate float64, L2REG float64 ) {
	// Make sure to set the gradients to zero.
	for o := 0; o < self.OutputDim; o++ {
		self.Biases[o] = self.Biases[o] - (learningRate/m) * self.GradientsBiases[o]
		self.GradientsBiases[o] = 0.0
	}
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			self.Weights[i][o] = (1-learningRate*L2REG) * self.Weights[i][o] - (learningRate/m) * self.GradientsWeights[i][o]
			self.GradientsWeights[i][o] = 0.0
		}
	}
}

func (self *BaseLayer) CalculateWeightsSum( power float64 ) float64 {
	output := 0.0
	for i := 0; i < self.InputDim; i++ {
		for o := 0; o < self.OutputDim; o++ {
			output += math.Pow(math.Abs(self.Weights[i][o]),power)
		}
	}
	return output
}

// This needs to be optimised.

func (self *BaseLayer) CalculateDeltaRegular( w [][]float64, d []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		tmpDelta := 0.0
		for n := 0; n < len(d); n++ {
			tmpDelta += w[o][n] * d[n]
		}
		// Hademard-product. Pointwise multiplication with the differential.
		self.Deltas[o] = tmpDelta * utils.SigmoidDifferential(self.Zetas[o])
	}
}

func (self *BaseLayer) CalculateDeltaOutputSquaredError( outputTruthVector []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		// Hademard-product. Pointwise multiplication with the differential.
		self.Deltas[o] = (self.Activations[o] - outputTruthVector[o]) * utils.SigmoidDifferential(self.Zetas[o])
	}
}

func (self *BaseLayer) CalculateDeltaOutputCrossEntropy( outputTruthVector []float64 ) {
	for o := 0; o < self.OutputDim; o++ {
		self.Deltas[o] = self.Activations[o] - outputTruthVector[o]
	}
}


