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

func initParams(inputSize int, hiddenSize int, outputSize int) (W1 [][]float64, b1 [][]float64, W2 [][]float64, b2 [][]float64) {
	W1 = randomMatrix(hiddenSize, inputSize, 0.5, -0.5)
	b1 = randomMatrix(hiddenSize, 1, 0.5, -0.5)
	W2 = randomMatrix(outputSize, hiddenSize, 0.5, -0.5)
	b2 = randomMatrix(outputSize, 1, 0.5, -0.5)
	return W1, b1, W2, b2
}

// tested
func ForwardProp(W1 [][]float64, b1 [][]float64, W2 [][]float64, b2 [][]float64, X [][]float64) (Z1 [][]float64, A1 [][]float64, Z2 [][]float64, A2 [][]float64) {
	Z1 = matrixAddVector(matrixMultiplication(W1, X), b1)
	A1 = sigmoidMatrix(Z1)
	Z2 = matrixAddVector(matrixMultiplication(W2, A1), b2)
	A2 = Z2
	return Z1, A1, Z2, A2
}

func backwardProp(Z1 [][]float64, A1 [][]float64, Z2 [][]float64, A2 [][]float64, W1 [][]float64, W2 [][]float64, X [][]float64, Y [][]float64) (dW1 [][]float64, db1 [][]float64, dW2 [][]float64, db2 [][]float64) {
	m := float64(len(X[0]))
	dZ2 := matrixScalarMultiplication(matrixSubstraction(A2, Y), 2.0)
	dW2 = matrixScalarMultiplication(matrixMultiplication(dZ2, matrixTranspose(A1)), 1.0/m)
	db2 = matrixScalarMultiplication(matrixSum(dZ2), 1.0/m)
	dZ1 := matrixScalarMultiplication(matrixMultiplicationByElement(matrixMultiplication(matrixTranspose(W2), dZ2), sigmoidDerivativeMatrix(Z1)), 2.0)
	dW1 = matrixScalarMultiplication(matrixMultiplication(dZ1, matrixTranspose(X)), 1.0/m)
	db1 = matrixScalarMultiplication(matrixSum(dZ1), 1.0/m)
	return dW1, db1, dW2, db2
}

// tested
func updateParams(W1 [][]float64, b1 [][]float64, W2 [][]float64, b2 [][]float64, dW1 [][]float64, db1 [][]float64, dW2 [][]float64, db2 [][]float64, learning_rate float64) (W1_updated [][]float64, b1_updated [][]float64, W2_updated [][]float64, b2_updated [][]float64) {
	W1_updated = matrixSubstraction(W1, matrixScalarMultiplication(dW1, learning_rate))
	b1_updated = matrixSubstraction(b1, matrixScalarMultiplication(db1, learning_rate))
	W2_updated = matrixSubstraction(W2, matrixScalarMultiplication(dW2, learning_rate))
	b2_updated = matrixSubstraction(b2, matrixScalarMultiplication(db2, learning_rate))
	return W1_updated, b1_updated, W2_updated, b2_updated
}

func getLoss(A2 [][]float64, Y [][]float64) float64 {
	m := float64(len(A2[0]))
	var sum float64
	for i := 0; i < len(A2); i++ {
		sum += math.Pow(A2[0][i]-Y[0][i], 2)
	}
	return sum / m
}

func GradiantDescent(X [][]float64, Y [][]float64, iterations int, learnRate float64, inputSize int, hiddenSize int, outputSize int) (W1 [][]float64, b1 [][]float64, W2 [][]float64, b2 [][]float64) {
	W1, b1, W2, b2 = initParams(inputSize, hiddenSize, outputSize)

	tenPercent := iterations / 10
	for i := 0; i < iterations; i++ {
		Z1, A1, Z2, A2 := ForwardProp(W1, b1, W2, b2, X)
		dW1, db1, dW2, db2 := backwardProp(Z1, A1, Z2, A2, W1, W2, X, Y)
		W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, learnRate)
		if i%tenPercent == 0 {
			fmt.Print("Progress: ", i/tenPercent, "/", 10, " Loss: ")
			fmt.Println(getLoss(A2, Y))
			util.WriteValuesToFile(A2, "./data/Out"+fmt.Sprint(i/tenPercent)+".txt")
		}
		if i == iterations-1 {
			fmt.Print("Progress: ", 10, "/", 10, " Loss: ")
			fmt.Println(getLoss(A2, Y))
		}
	}
	return W1, b1, W2, b2
}
