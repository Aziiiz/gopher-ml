package common

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func SumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()
	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis must be 0 or 1")
	}
	return output, nil
}

// sigmoid more .. https://mathworld.wolfram.com/SigmoidFunction.html
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime
func SigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}
