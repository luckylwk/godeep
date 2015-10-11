package utils

//

import (
	"math"
)

//

type CostFunction interface {
    Name() string;
    Fn( a []float64, y []float64 ) float64;
    FnPrime( a []float64, y []float64 ) float64;
}


type CostSquaredError struct {}

func (self *CostSquaredError) Name() string {
    return "Squared Error"
}

func (self *CostSquaredError) Fn( a []float64, y []float64 ) float64 {
	cost := 0.0
	for o := 0; o < len(a); o++ {
		thisNeuronError := a[o] - y[o]
		cost += 0.5 * math.Pow(thisNeuronError, 2.0)
	}
	return cost
}



type CostCrossEntropy struct {}

func (self *CostCrossEntropy) Name() string {
    return "Cross-Entropy (Multinomial Logistic)"
}

func (self *CostCrossEntropy) Fn( a []float64, y []float64 ) float64 {
	cost := 0.0
	for o := 0; o < len(a); o++ {
		cost += -1.0 * (y[o]*math.Log(a[o]) + (1.0-y[o])*math.Log(1.0-a[o]) )
	}
	return cost
}

