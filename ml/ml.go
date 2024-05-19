package ml

import (
	"Mitschl/Gomal/util"
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

func ForwardPropMultiLayer(weights [][][]float64, biases [][][]float64, input [][]float64) (sums [][][]float64, outputs [][][]float64) {
	sums = make([][][]float64, 0)
	outputs = make([][][]float64, 0)

	sums = append(sums, matrixAddVector(matrixMultiplication(weights[0], input), biases[0]))
	outputs = append(outputs, sigmoidMatrix(sums[0]))

	for i := 1; i < len(weights); i++ {
		sums = append(sums, matrixAddVector(matrixMultiplication(weights[i], outputs[i-1]), biases[i]))
		outputs = append(outputs, sums[i])
	}

	return sums, outputs
}

func backwardProp(sum1 [][]float64, output1 [][]float64, sum2 [][]float64, output2 [][]float64, weight1 [][]float64, weight2 [][]float64, input [][]float64, expected [][]float64) (dWeight1 [][]float64, dBias1 [][]float64, dWeight2 [][]float64, dBias2 [][]float64) {
	m := float64(len(input[0]))
	dZ2 := matrixScalarMultiplication(matrixSubstraction(output2, expected), 2.0)
	dWeight2 = matrixScalarMultiplication(matrixMultiplication(dZ2, matrixTranspose(output1)), 1.0/m)
	dBias2 = matrixScalarMultiplication(matrixSum(dZ2), 1.0/m)
	dZ1 := matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(weight2), dZ2), sigmoidDerivativeMatrix(sum1)), 2.0)
	dWeight1 = matrixScalarMultiplication(matrixMultiplication(dZ1, matrixTranspose(input)), 1.0/m)
	dBias1 = matrixScalarMultiplication(matrixSum(dZ1), 1.0/m)
	return dWeight1, dBias1, dWeight2, dBias2
}

func backwardPropMultiLayer(sums [][][]float64, outputs [][][]float64, weights [][][]float64, input [][]float64, expected [][]float64) (dWeights [][][]float64, dBias [][][]float64) {
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
	dZ[len(dZ)-1] = matrixScalarMultiplication(matrixSubstraction(outputs[len(outputs)-1], expected), 2.0)
	dWeights[len(dWeights)-1] = matrixScalarMultiplication(matrixMultiplication(dZ[len(dZ)-1], matrixTranspose(outputs[len(outputs)-2])), 1.0/m)
	dBias[len(dBias)-1] = matrixScalarMultiplication(matrixSum(dZ[len(dZ)-1]), 1.0/m)

	for i := len(dZ) - 2; i > 0; i-- {
		dZ[i] = matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(weights[i+1]), dZ[i+1]), sigmoidDerivativeMatrix(sums[i])), 2.0)
		dWeights[i] = matrixScalarMultiplication(matrixMultiplication(dZ[i], matrixTranspose(outputs[i-1])), 1.0/m)
		dBias[i] = matrixScalarMultiplication(matrixSum(dZ[i]), 1.0/m)
	}
	dZ[0] = matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(weights[1]), dZ[1]), sigmoidDerivativeMatrix(sums[0])), 2.0)
	dWeights[0] = matrixScalarMultiplication(matrixMultiplication(dZ[0], matrixTranspose(input)), 1.0/m)
	dBias[0] = matrixScalarMultiplication(matrixSum(dZ[0]), 1.0/m) 

	return dWeights, dBias
}

// tested
func updateParams(weight1 [][]float64, bias1 [][]float64, weight2 [][]float64, bias2 [][]float64, dWeight1 [][]float64, dBias1 [][]float64, dWeight2 [][]float64, dBias2 [][]float64, learningRate float64) (weight1Updated [][]float64, bias1Updated [][]float64, weight2Updated [][]float64, bias2Updated [][]float64) {
	weight1Updated = matrixSubstraction(weight1, matrixScalarMultiplication(dWeight1, learningRate))
	bias1Updated = matrixSubstraction(bias1, matrixScalarMultiplication(dBias1, learningRate))
	weight2Updated = matrixSubstraction(weight2, matrixScalarMultiplication(dWeight2, learningRate))
	bias2Updated = matrixSubstraction(bias2, matrixScalarMultiplication(dBias2, learningRate))
	return weight1Updated, bias1Updated, weight2Updated, bias2Updated
}

func updateParamsMultiLayer(weights [][][]float64, biases [][][]float64, dWeights [][][]float64, dBias [][][]float64, learningRate float64) (weightsUpdated [][][]float64, biasesUpdated [][][]float64) {
	weightsUpdated = make([][][]float64, 0)
	biasesUpdated = make([][][]float64, 0)

	for i := 0; i < len(weights); i++ {
		weightsUpdated = append(weightsUpdated, matrixSubstraction(weights[i], matrixScalarMultiplication(dWeights[i], learningRate)))
		biasesUpdated = append(biasesUpdated, matrixSubstraction(biases[i], matrixScalarMultiplication(dBias[i], learningRate)))
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

func GradiantDescent(input [][]float64, expected [][]float64, iterations int, learnRate float64, inputSize int, hiddenSizes []int, outputSize int) (weights [][][]float64, biases [][][]float64) {
	weights, biases = initParams(inputSize, hiddenSizes, outputSize)

	tenPercent := iterations / 10
	for i := 0; i < iterations; i++ {
		sums, outputs := ForwardPropMultiLayer(weights, biases, input)
		//dW1, db1, dW2, db2 := backwardProp(sums[0], outputs[0], sums[1], outputs[1], weights[0], weights[1], input, expected)
		//weights, biases = updateParamsMultiLayer(weights, biases, [][][]float64{dW1, dW2}, [][][]float64{db1, db2}, learnRate)
		dWeights, dBias := backwardPropMultiLayer(sums, outputs, weights, input, expected)
		weights, biases = updateParamsMultiLayer(weights, biases, dWeights, dBias, learnRate)
		if i%tenPercent == 0 {
			fmt.Print("Progress: ", i/tenPercent, "/", 10, " Loss: ")
			fmt.Println(getLoss(outputs[1], expected))
			util.WriteValuesToFile(outputs[1], "./data/Out"+fmt.Sprint(i/tenPercent)+".txt")
		}
		if i == iterations-1 {
			fmt.Print("Progress: ", 10, "/", 10, " Loss: ")
			fmt.Println(getLoss(outputs[1], expected))
		}
	}
	return weights, biases
}
