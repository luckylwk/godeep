package utils

//

import (
	"math"
)

//

func Sigmoid(zeta float64) float64 {
    return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) )
}

func SigmoidDifferential(zeta float64) float64 {
    sigmoid := Sigmoid(zeta)
    return sigmoid * (1.0 - sigmoid)
}


func Tanh(zeta float64) float64 {
    return math.Tanh(zeta)
}

func TanhDifferential(zeta float64) float64 {
    tanh := Tanh(zeta)
    return 1.0 - tanh * tanh
}


func ReLU(zeta float64) float64 {
    return math.Max(0,zeta)
}

func ReLUDifferential(zeta float64) float64 {
    output := 0.0
    if zeta > 0.0 {
    	output = zeta
    }
    return output
}


func SoftPlus(zeta float64) float64 {
    return math.Log(1.0 + math.Pow(math.E, float64(zeta)))
}

func SoftPlusDifferential(zeta float64) float64 {
    return Sigmoid(zeta)
}
