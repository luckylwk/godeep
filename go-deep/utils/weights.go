package utils

//

import (
	"math"
	"math/rand"
)

//

func WeightsUniformLeCun( dim1 int, dim2 int ) [][]float64 {
	// Reference: LeCun 98, Efficient Backprop
	// http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
	scale := math.Sqrt( 3.0 / float64(dim1) )
	return WeightsUniform( dim1, dim2, scale )
}

func WeightsUniformHe( dim1 int, dim2 int ) [][]float64 {
	scale := math.Sqrt( 6.0 / float64(dim1) )
	return WeightsUniform( dim1, dim2, scale )
}

func WeightsUniformGlorot( dim1 int, dim2 int ) [][]float64 {
	scale := math.Sqrt( 6.0 / float64(dim1+dim2) )
	return WeightsUniform( dim1, dim2, scale )
}

func WeightsNormalGlorot( dim1 int, dim2 int ) [][]float64 {
	// Reference: Glorot & Bengio, AISTATS 2010
	scale := math.Sqrt( 2.0 / float64(dim1+dim2) )
	return WeightsNormal( dim1, dim2, scale )
}

func WeightsNormalHe( dim1 int, dim2 int ) [][]float64 {
	// Reference:  He et al., http://arxiv.org/abs/1502.01852
	scale := math.Sqrt( 2.0 / float64(dim1) )
	return WeightsNormal( dim1, dim2, scale )
}

func WeightsUniform( dim1 int, dim2 int, scale float64 ) [][]float64 {
	matrix := make([][]float64, dim1)
	for i := 0; i < dim1; i++ {
		matrix[i] = make([]float64, dim2)
		for o := 0; o < dim2; o++ {
			matrix[i][o] = ( rand.Float64() * 2 * scale - scale )
		}
	}
	return matrix
}


func WeightsNormal( dim1 int, dim2 int, scale float64 ) [][]float64 {
	matrix := make([][]float64, dim1)
	for i := 0; i < dim1; i++ {
		matrix[i] = make([]float64, dim2)
		for o := 0; o < dim2; o++ {
			matrix[i][o] = ( rand.NormFloat64() * 2 * scale - scale )
		}
	}
	return matrix
}






func MatrixConstant( dim1 int, dim2 int, value float64 ) [][]float64 {
	// Note: make initialises to zero by default.
	matrix := make([][]float64, dim1)
	for i := 0; i < dim1; i++ {
		matrix[i] = make([]float64, dim2)
		if value != 0.0 {
			for j := 0; j < dim2; j++ {
				matrix[i][j] = value
			}
		}
	}
	return matrix
}


func VectorConstant( dim1 int, value float64 ) []float64 {
	// Note: make initialises to zero by default.
	vector := make([]float64, dim1)
	if value != 0.0 {
		for i := 0; i < dim1; i++ {
			vector[i] = value
		}
	}
	return vector
}

