package utils

//

import (
	"math"
)

//

type ActivationFunction interface {
    Name() string
    Fn( zeta float64 ) float64
    FnPrime( zeta float64 ) float64
}


type Sigmoid struct {}

func (self *Sigmoid) Name() string {
    return "Sigmoid / Logistic function"
}
func (self *Sigmoid) Fn( zeta float64 ) float64 {
    return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) )
}
func (self *Sigmoid) FnPrime( zeta float64 ) float64 {
    sigmoid := self.Fn(zeta)
    return sigmoid * (1.0 - sigmoid)
}


type Tanh struct {}

func (self *Tanh) Name() string {
    return "Tanh / Hyperbolic Tangent"
}
func (self *Tanh) Fn( zeta float64 ) float64 {
    return math.Tanh(zeta)
}
func (self *Tanh) FnPrime( zeta float64 ) float64 {
    tanh := self.Fn(zeta)
    return 1.0 - tanh * tanh
}


type ReLU struct {}

func (self *ReLU) Name() string {
    return "ReLU / Rectified Linear Unit"
}
func (self *ReLU) Fn( zeta float64 ) float64 {
    return math.Max(0,zeta)
}
func (self *ReLU) FnPrime( zeta float64 ) float64 {
    output := 0.0
    if zeta > 0.0 {
        output = zeta
    }
    return output
}


type SoftPlus struct {}

func (self *SoftPlus) Name() string {
    return "SoftPlus"
}
func (self *SoftPlus) Fn( zeta float64 ) float64 {
    return math.Log(1.0 + math.Pow(math.E, float64(zeta)))
}
func (self *SoftPlus) FnPrime( zeta float64 ) float64 {
    return 1.0 / ( 1.0 + math.Pow(math.E, -float64(zeta)) ) // Sigmoid!
}
