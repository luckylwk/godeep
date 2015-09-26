package utils

//

import (
	// "math/rand"
)

//

func Argmax(A []float64) int{
    var x int
    v := -1.0
    for i := 0; i < len(A); i++ {
        if A[i] > v {
            v = A[i]
            x = i
        }
    }
    return x
}

