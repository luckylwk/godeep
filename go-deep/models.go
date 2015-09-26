package godeep

//

import (
	"fmt"
	"github.com/luckylwk/godeep/go-deep/layers"
	"github.com/luckylwk/godeep/go-deep/utils"
	"math"
	"math/rand"
)

//

type Sequential struct {
	// Dimensions of each input sample.
	inputDim int
	// The model holds pointers to all the layers
	layers []*layers.BaseLayer
	numLayers int
	//
	costFunction string
}

func (self *Sequential) Init( inputDim int, costFunction string ) {
	self.inputDim = inputDim
	self.costFunction = costFunction
}

func (self *Sequential) Add( newLayer *layers.BaseLayer ) {
	self.layers = append(self.layers, newLayer)
}

func (self *Sequential) Compile() {
	// Get the total number of layers to work with.
	self.numLayers = len(self.layers)
	// Loop over the layers and connect the input and output dimensions.
	// Also have each layer initialise the parameters needed to run.
	for l := 0; l < self.numLayers; l++ {
		// If it is the first layer it needs to be connected to the input data size.
		if l == 0 {
			self.layers[l].InputDim = self.inputDim
		} else {
			self.layers[l].InputDim = self.layers[l-1].OutputDim
		}
		self.layers[l].InitParameters()
	}
}

func (self *Sequential) FeedForward( inputVector []float64 ) {
	// Take input data. Always use vector
	for l := 0; l < self.numLayers; l++ {
		if l == 0 {
			self.layers[l].FeedForward( inputVector )
		} else {
			// Take the output of the previous layer as the input vector.
			self.layers[l].FeedForward( self.layers[l-1].Activations )
		}
	}
}

func (self *Sequential) CalculateCost( outputTruthVector []float64, L2REG float64 ) float64 {
	// What loss function to use?
	cost := 0.0 //initial error is zero
	// Squared Error.
	if self.costFunction == "SE" {
		// For each activation in the output layer we check how it relates to the actual value.
		for o := 0; o < self.layers[self.numLayers-1].OutputDim; o++ {
			thisNeuronError := self.layers[self.numLayers-1].Activations[o] - outputTruthVector[o]
			cost += 0.5 * math.Pow(thisNeuronError, 2.0)
		}
	}
	if self.costFunction == "CROSSENTROPY" {
		for o := 0; o < self.layers[self.numLayers-1].OutputDim; o++ {
			thisNeuron := self.layers[self.numLayers-1].Activations[o]
			cost += -1.0 * (outputTruthVector[o]*math.Log(thisNeuron) + (1.0-outputTruthVector[o])*math.Log(1.0-thisNeuron) )
		}
	}
	// Regularisation.
	// penaltyL1 += 0.5 * L1REG * n.calcWeightsPenalty(1.0)
	penaltyL2 := 0.0
	if L2REG != 0.0 {
		for l := 0; l < self.numLayers; l++ {
			penaltyL2 += 0.5 * L2REG * self.layers[l].CalculateWeightsSum(2.0)
		}
	}
	return cost + penaltyL2
}

func (self *Sequential) BackPropagate( outputTruthVector []float64, batchSize int, doUpdate bool, L2REG float64 ) {
	// Start at the final layer and propagate backwards.
	for l := self.numLayers-1; l >= 0; l-- {
		if l == self.numLayers-1 {
			// This is the output layer: Calculate the deltaC/deltaZ
			if self.costFunction == "SE" {
				self.layers[l].CalculateDeltaOutputSquaredError(outputTruthVector)
			}
			if self.costFunction == "CROSSENTROPY" {
				self.layers[l].CalculateDeltaOutputCrossEntropy(outputTruthVector)
			}
		} else {
			// Needs the weights and delta's of the next layer.
			w := self.layers[l+1].Weights
			d := self.layers[l+1].Deltas
			self.layers[l].CalculateDeltaRegular(w,d)
		}
		self.layers[l].CalculateGradients()
		if doUpdate {
			self.layers[l].UpdateWeights( float64(batchSize), 0.3, L2REG )
		}
	}
}

func (self *Sequential) Train( data [][]float64, labels [][]float64, epochs int, batchSize int ) {
	//
	fmt.Println("\n\n *** TRAINING...")
	// Set the regularisation parameter.
	L2REG := 0.0
	// Train for all epochs.
	for e := 0; e < epochs; e++ {
		// Create a random permutation of the data.
		randomPerm := rand.Perm(len(data))
		//
		error := 0.0
		accuracy := 0.0
		// Loop over the dataset.
		for d := 0; d < len(data); d++ {
			self.FeedForward(data[randomPerm[d]])
			error += self.CalculateCost(labels[randomPerm[d]],L2REG)
			self.BackPropagate(labels[randomPerm[d]], batchSize, (d+1)%batchSize==0, L2REG )
			if utils.Argmax(self.layers[self.numLayers-1].Activations) == utils.Argmax(labels[randomPerm[d]]) {
				accuracy++
			}
			if (d+1) % 500 == 0 {
				fmt.Printf("\tEpoch %vth / Progress: %.2f%% / Avg. Loss: %.4f / Accuracy %.2f%% \r", e+1, float64(d+1)*100/float64(len(data)), error/float64(d+1), 100.0*accuracy/float64(d+1) )
			}
		}
		fmt.Printf("\r\n")
	}
}

func (self *Sequential) Predict( data [][]float64, labels [][]float64 ) {
	fmt.Println(" *** PREDICTING TEST-DATA")
	accuracy := 0.0
	for d := 0; d < len(data); d++ {
		self.FeedForward(data[d])
		if utils.Argmax(self.layers[self.numLayers-1].Activations) == utils.Argmax(labels[d]) {
			accuracy++
		}
		if (d+1) % 100 == 0 {
			fmt.Printf("\tProgress: %.2f%% / Accuracy %.2f%% \r", float64(d+1)*100/float64(len(data)), 100.0*accuracy/float64(d+1) )
		}
	}

}


