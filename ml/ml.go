package ml

import (
	"fmt"
	"math"
	"math/rand"
)

// tested
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// tested
func sigmoidMatrix(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m); i++ {
		var row []float64
		for j := 0; j < len(m[0]); j++ {
			row = append(row, sigmoid(m[i][j]))
		}
		result = append(result, row)
	}
	return result
}

// tested
func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// tested
func sigmoidDerivativeMatrix(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m); i++ {
		var row []float64
		for j := 0; j < len(m[0]); j++ {
			row = append(row, sigmoidDerivative(m[i][j]))
		}
		result = append(result, row)
	}
	return result
}

func linear(x float64) float64 {
	return x
}

func linearMatrix(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m); i++ {
		var row []float64
		for j := 0; j < len(m[0]); j++ {
			row = append(row, linear(m[i][j]))
		}
		result = append(result, row)
	}
	return result
}

func linearDerivative(x float64) float64 {
	return 1
}

func linearDerivativeMatrix(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m); i++ {
		var row []float64
		for j := 0; j < len(m[0]); j++ {
			row = append(row, linearDerivative(m[i][j]))
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixMultiplication(a [][]float64, b [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(a); i++ {
		var row []float64
		for j := 0; j < len(b[0]); j++ {
			var sum float64
			for k := 0; k < len(b); k++ {
				sum += a[i][k] * b[k][j]
			}
			row = append(row, sum)
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixAddVector(a [][]float64, b [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(a); i++ {
		var row []float64
		for j := 0; j < len(a[0]); j++ {
			row = append(row, a[i][j]+b[i][0])
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixMultiplicationByElement(a [][]float64, b [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(a); i++ {
		var row []float64
		for j := 0; j < len(a[0]); j++ {
			row = append(row, a[i][j]*b[i][j])
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixSubstraction(a [][]float64, b [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(a); i++ {
		var row []float64
		for j := 0; j < len(a[0]); j++ {
			row = append(row, a[i][j]-b[i][j])
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixTranspose(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m[0]); i++ {
		var row []float64
		for j := 0; j < len(m); j++ {
			row = append(row, m[j][i])
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixScalarMultiplication(a [][]float64, b float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(a); i++ {
		var row []float64
		for j := 0; j < len(a[0]); j++ {
			row = append(row, a[i][j]*b)
		}
		result = append(result, row)
	}
	return result
}

// tested
func matrixSum(m [][]float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < len(m); i++ {
		var row []float64
		var sum float64
		for j := 0; j < len(m[0]); j++ {
			sum += m[i][j]
		}
		row = append(row, sum)
		result = append(result, row)
	}
	return result
}

// tested
func randomMatrix(rows int, cols int, max float64, min float64) (c [][]float64) {
	var result [][]float64
	for i := 0; i < rows; i++ {
		var row []float64
		for j := 0; j < cols; j++ {
			row = append(row, rand.Float64()*(max-min)+min)
		}
		result = append(result, row)
	}
	return result
}

func initParams(inputSize int, hiddenSizes []int, outputSize int) (weights [][][]float64, biases [][][]float64) {
	weights = make([][][]float64, 0)
	biases = make([][][]float64, 0)

	// weight/bias for input to first hidden layer
	weights = append(weights, randomMatrix(hiddenSizes[0], inputSize, 0.5, -0.5))
	biases = append(biases, randomMatrix(hiddenSizes[0], 1, 0.5, -0.5))

	// weight/bias for hidden layers
	for i := 0; i < len(hiddenSizes)-1; i++ {
		weights = append(weights, randomMatrix(hiddenSizes[i+1], hiddenSizes[i], 0.5, -0.5))
		biases = append(biases, randomMatrix(hiddenSizes[i+1], 1, 0.5, -0.5))
	}

	// weight/bias for last hidden layer to output
	weights = append(weights, randomMatrix(outputSize, hiddenSizes[len(hiddenSizes)-1], 0.5, -0.5))
	biases = append(biases, randomMatrix(outputSize, 1, 0.5, -0.5))

	return weights, biases
}

// tested
func ForwardProp(weight1 [][]float64, bias1 [][]float64, weight2 [][]float64, bias2 [][]float64, input [][]float64) (sum1 [][]float64, output1 [][]float64, sum2 [][]float64, output2 [][]float64) {
	sum1 = matrixAddVector(matrixMultiplication(weight1, input), bias1)
	output1 = sigmoidMatrix(sum1)
	sum2 = matrixAddVector(matrixMultiplication(weight2, output1), bias2)
	output2 = sum2

	return sum1, output1, sum2, output2
}

func ForwardPropMultiLayer(weights [][][]float64, biases [][][]float64, activationFunctions []string, input [][]float64) (sums [][][]float64, outputs [][][]float64) {
	sums = make([][][]float64, 0)
	outputs = make([][][]float64, 0)

	sums = append(sums, matrixAddVector(matrixMultiplication(weights[0], input), biases[0]))
	if activationFunctions[0] == "sigmoid" {
		outputs = append(outputs, sigmoidMatrix(sums[0]))
	} else if activationFunctions[0] == "linear" {
		outputs = append(outputs, linearMatrix(sums[0]))
	} else {
		fmt.Println("Activation function not supported")
		return nil, nil
	}

	for i := 1; i < len(weights); i++ {
		sums = append(sums, matrixAddVector(matrixMultiplication(weights[i], outputs[i-1]), biases[i]))
		if activationFunctions[i] == "sigmoid" {
			outputs = append(outputs, sigmoidMatrix(sums[i]))
		} else if activationFunctions[i] == "linear" {
			outputs = append(outputs, linearMatrix(sums[i]))
		} else {
			fmt.Println("Activation function not supported")
			return nil, nil
		}
	}

	return sums, outputs
}

func backwardPropMultiLayer(sums [][][]float64, outputs [][][]float64, weights [][][]float64, activationFunctions []string, input [][]float64, expected [][]float64) (dWeights [][][]float64, dBias [][][]float64) {
	m := float64(len(input[0]))
	dZ := make([][][]float64, len(outputs))
	for i := 0; i < len(dZ); i++ {
		dZ[i] = make([][]float64, len(outputs[i]))
		for j := 0; j < len(dZ[i]); j++ {
			dZ[i][j] = make([]float64, len(outputs[i][0]))
		}
	}
	dWeights = make([][][]float64, len(weights))
	for i := 0; i < len(dWeights); i++ {
		dWeights[i] = make([][]float64, len(weights[i]))
		for j := 0; j < len(dWeights[i]); j++ {
			dWeights[i][j] = make([]float64, len(weights[i][0]))
		}
	}
	dBias = make([][][]float64, len(outputs))
	for i := 0; i < len(dBias); i++ {
		dBias[i] = make([][]float64, len(outputs[i]))
		for j := 0; j < len(dBias[i]); j++ {
			dBias[i][j] = make([]float64, 1)
		}
	}

	var derivative_output [][]float64
	if activationFunctions[len(activationFunctions)-1] == "sigmoid" {
		derivative_output = sigmoidDerivativeMatrix(sums[len(sums)-1])
	} else if activationFunctions[len(activationFunctions)-1] == "linear" {
		derivative_output = linearDerivativeMatrix(sums[len(sums)-1])
	} else {
		fmt.Println("Activation function not supported")
		return nil, nil
	}
	dZ[len(dZ)-1] = matrixScalarMultiplication(matrixMultiplicationByElement(matrixSubstraction(outputs[len(outputs)-1], expected), derivative_output), 2.0)
	dWeights[len(dWeights)-1] = matrixScalarMultiplication(matrixMultiplication(dZ[len(dZ)-1], matrixTranspose(outputs[len(outputs)-2])), 1.0/m)
	dBias[len(dBias)-1] = matrixScalarMultiplication(matrixSum(dZ[len(dZ)-1]), 1.0/m)

	for i := len(dZ) - 2; i > 0; i-- {
		var derivative [][]float64
		if activationFunctions[i] == "sigmoid" {
			derivative = sigmoidDerivativeMatrix(sums[i])
		} else if activationFunctions[i] == "linear" {
			derivative = linearDerivativeMatrix(sums[i])
		} else {
			fmt.Println("Activation function not supported")
			return nil, nil
		}
		dZ[i] = matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(weights[i+1]), dZ[i+1]), derivative), 2.0)
		dWeights[i] = matrixScalarMultiplication(matrixMultiplication(dZ[i], matrixTranspose(outputs[i-1])), 1.0/m)
		dBias[i] = matrixScalarMultiplication(matrixSum(dZ[i]), 1.0/m)
	}

	var derivative_input [][]float64
	if activationFunctions[0] == "sigmoid" {
		derivative_input = sigmoidDerivativeMatrix(sums[0])
	} else if activationFunctions[0] == "linear" {
		derivative_input = linearDerivativeMatrix(sums[0])
	} else {
		fmt.Println("Activation function not supported")
		return nil, nil
	}
	dZ[0] = matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(weights[1]), dZ[1]), derivative_input), 2.0)
	dWeights[0] = matrixScalarMultiplication(matrixMultiplication(dZ[0], matrixTranspose(input)), 1.0/m)
	dBias[0] = matrixScalarMultiplication(matrixSum(dZ[0]), 1.0/m) 

	return dWeights, dBias
}

// tested
func updateParamsMultiLayer(weights [][][]float64, biases [][][]float64, dWeights [][][]float64, lastDWeights [][][]float64, dBias [][][]float64, lastDBias [][][]float64, learningRate float64, momentumFactor float64, learningRateDecay float64, iteration int) (weightsUpdated [][][]float64, biasesUpdated [][][]float64) {
	weightsUpdated = make([][][]float64, 0)
	biasesUpdated = make([][][]float64, 0)

	temp_learnRate := learningRate * math.Exp(-learningRateDecay*float64(iteration))

	for i := 0; i < len(weights); i++ {
		dW := matrixAddVector(matrixScalarMultiplication(dWeights[i], temp_learnRate), matrixScalarMultiplication(lastDWeights[i], momentumFactor))
		weightsUpdated = append(weightsUpdated, matrixSubstraction(weights[i], dW))
		dB := matrixAddVector(matrixScalarMultiplication(dBias[i], temp_learnRate), matrixScalarMultiplication(lastDBias[i], momentumFactor))
		biasesUpdated = append(biasesUpdated, matrixSubstraction(biases[i], dB))
	}

	return weightsUpdated, biasesUpdated
}

func getLoss(output [][]float64, expected [][]float64) float64 {
	m := float64(len(output[0]))
	var sum float64
	for i := 0; i < len(output); i++ {
		sum += math.Pow(output[0][i]-expected[0][i], 2)
	}
	return sum / m
}

func GradiantDescent(input [][]float64, expected [][]float64, iterations int, learnRate float64, momentumFactor float64, learingRateDecay float64, inputSize int, hiddenSizes []int, outputSize int, activationFunctions []string) (weights [][][]float64, biases [][][]float64, losses []float64) {
	weights, biases = initParams(inputSize, hiddenSizes, outputSize)

	fivePercent := iterations / 20
	onePercent := iterations / 100
	lastDWeights := make([][][]float64, len(weights))
	for i := 0; i < len(lastDWeights); i++ {
		lastDWeights[i] = make([][]float64, len(weights[i]))
		for j := 0; j < len(lastDWeights[i]); j++ {
			lastDWeights[i][j] = make([]float64, len(weights[i][0]))
		}
	}
	lastDBias := make([][][]float64, len(biases))
	for i := 0; i < len(lastDBias); i++ {
		lastDBias[i] = make([][]float64, len(biases[i]))
		for j := 0; j < len(lastDBias[i]); j++ {
			lastDBias[i][j] = make([]float64, 1)
		}
	}

	for i := 0; i < iterations; i++ {
		sums, outputs := ForwardPropMultiLayer(weights, biases, activationFunctions, input)
		dWeights, dBias := backwardPropMultiLayer(sums, outputs, weights, activationFunctions, input, expected)
		weights, biases = updateParamsMultiLayer(weights, biases, dWeights, lastDWeights, dBias, lastDBias, learnRate, momentumFactor, learingRateDecay, i)
		lastDWeights = dWeights
		lastDBias = dBias
		if i%onePercent == 0 {
			loss := getLoss(outputs[len(outputs)-1], expected)
			losses = append(losses, loss)
			if i%fivePercent == 0 {
				fmt.Println("Progress: ", i/fivePercent, "/", 20, " Loss: ", loss)
			}
		}
	}
	return weights, biases, losses
}
