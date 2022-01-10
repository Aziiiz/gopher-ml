package main

import (
	"encoding/csv"
	"errors"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"log"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet struct has all info to train over the network
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// architercture on the network
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// newNetwork inits new newtwork.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// sigmoid more .. https://mathworld.wolfram.com/SigmoidFunction.html
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

// train dataset
func (nn *neuralNet) train(x, y *mat.Dense) error {

	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons,
		nn.config.hiddenNeurons, nil)

	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)

	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)

	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// define output network
	output := new(mat.Dense)

	// by using backpropagation measure weight
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// define learned network
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil

}

// backpropagation method for distrubution
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// training model for some epoch by using backpropagtion
	for i := 0; i < nn.config.numEpochs; i++ {

		//finish up process of direct propagation
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBout := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		outputLayerInput.Apply(addBout, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		//finish up distrubution
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrime(v)
		}

		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}

		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}
	return nil
}

//sum matrixes
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

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
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// predic function predicts from trained network
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// define output
	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 {
		return v + nn.bHidden.At(0, col)
	}

	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySimoid := func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}

	hiddenLayerActivations.Apply(applySimoid, hiddenLayerActivations)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBout := func(_, col int, v float64) float64 {
		return v + nn.bOut.At(0, col)
	}

	outputLayerInput.Apply(addBout, outputLayerInput)
	output.Apply(applySimoid, outputLayerInput)

	return output, nil
}

func main() {

	f, err := os.Open("data/train.csv")
	if err != nil {
		log.Error(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Error(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsDate := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {

		if idx == 0 {
			continue
		}

		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Error(err)
			}

			if i == 4 || i == 5 || i == 6 {
				labelsDate[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}
		}
	}
}
