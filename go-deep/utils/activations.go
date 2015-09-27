package utils

//

import (
	"math"
)

//

type ActivationFunction interface {
    Name() string;
    Fn( zeta float64 ) float64;
    FnPrime( zeta float64 ) float64;
}


type Sigmoid struct {}

func (f *Sigmoid) Name() string {
    return "Sigmoid / Logistic function"
}
func (f *Sigmoid) Fn( zeta float64 ) float64 {
    return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) )
}
func (f *Sigmoid) FnPrime( zeta float64 ) float64 {
    sigmoid := f.Fn(zeta)
    return sigmoid * (1.0 - sigmoid)
}


type Tanh struct {}

func (f *Tanh) Name() string {
    return "Tanh / Hyperbolic Tangent"
}
func (f *Tanh) Fn( zeta float64 ) float64 {
    return math.Tanh(zeta)
}
func (f *Tanh) FnPrime( zeta float64 ) float64 {
    tanh := f.Fn(zeta)
    return 1.0 - tanh * tanh
}


type ReLU struct {}

func (f *ReLU) Name() string {
    return "ReLU / Rectified Linear Unit"
}
func (f *ReLU) Fn( zeta float64 ) float64 {
    return math.Max(0,zeta)
}
func (f *ReLU) FnPrime( zeta float64 ) float64 {
    output := 0.0
    if zeta > 0.0 {
        output = zeta
    }
    return output
}


type SoftPlus struct {}

func (f *SoftPlus) Name() string {
    return "SoftPlus"
}
func (f *SoftPlus) Fn( zeta float64 ) float64 {
    return math.Log(1.0 + math.Pow(math.E, float64(zeta)))
}
func (f *SoftPlus) FnPrime( zeta float64 ) float64 {
    return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) ) // Sigmoid!
}



// func SigmoidLogistic(zeta float64) float64 {
//     return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) )
// }

