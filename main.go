package main

import (
	"Mitschl/Gomal/ml"
	"math"
	"time"

	//"Mitschl/Gomal/util"
	"fmt"
	//"math"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type LearnRequestBody struct {
	Input  int   `json:"input"`
	Hidden []int `json:"hidden"`
	Output int   `json:"output"`
	ActivationFunctions []string `json:"activationFunctions"`

	X [][]float64 `json:"x"`
	Y [][]float64 `json:"y"`

	Iterations   int     `json:"iterations"`
	LearningRate float64 `json:"learningRate"`
	MomentumFactor float64 `json:"momentumFactor"`
	LearningRateDecay float64 `json:"learningRateDecay"`
}

type TrainResponseBody struct {
	Weights [][][]float64 `json:"weights"`
	Biases  [][][]float64   `json:"biases"`
	Losses []float64       `json:"losses"`
	Predictions [][]float64 `json:"predictions"`
}

func main() {

	r := gin.Default()
	config := cors.DefaultConfig()
    config.AllowAllOrigins = true
    config.AllowMethods = []string{"POST", "GET", "PUT", "OPTIONS"}
    config.AllowHeaders = []string{"Origin", "Content-Type", "Authorization", "Accept", "User-Agent", "Cache-Control", "Pragma"}
    config.ExposeHeaders = []string{"Content-Length"}
    config.AllowCredentials = true
    config.MaxAge = 12 * time.Hour

    r.Use(cors.New(config))

    r.LoadHTMLGlob("templ/*")

	r.POST("/train", func(c *gin.Context) {
		var requestBody LearnRequestBody
		err := c.BindJSON(&requestBody)

		if err != nil {
			fmt.Println(err.Error())
		}

		if requestBody.X == nil {
			c.JSON(400, gin.H{"error": "X is required"})
			return
		}
		if requestBody.Y == nil {
			c.JSON(400, gin.H{"error": "Y is required"})
			return
		}
		if requestBody.Iterations == 0 {
			c.JSON(400, gin.H{"error": "Iterations cant be 0"})
			return
		}
		if requestBody.LearningRate <= 0 {
			c.JSON(400, gin.H{"error": "LearningRate cant be 0 or smaller"})
			return
		}
		if requestBody.Input <= 0 {
			c.JSON(400, gin.H{"error": "Input cant smaller than 1"})
			return
		}
		if requestBody.Output <= 0 {
			c.JSON(400, gin.H{"error": "Output cant smaller than 1"})
			return
		}
		if len(requestBody.Hidden) == 0 {
			c.JSON(400, gin.H{"error": "Hidden Layers cant be empty"})
			return
		}
		for _, h := range requestBody.Hidden {
			if h <= 0 {
				c.JSON(400, gin.H{"error": "Hidden cant be smaller than 1"})
				return
			}
		}
		// check if X have the correct length
		if len(requestBody.X) != requestBody.Input {
			c.JSON(400, gin.H{"error": "X layer is not correct"})
			return
		}
		
		// check if Y have the correct length
		if len(requestBody.Y) != requestBody.Output {
			c.JSON(400, gin.H{"error": "Y layer is not correct"})
			return
		}

		if len(requestBody.ActivationFunctions) != len(requestBody.Hidden) + 1 {
			c.JSON(400, gin.H{"error": "Activation functions are not correct"})
			return
		}

		numExamples := len(requestBody.X[0])
		for i := 1; i < len(requestBody.X); i++ {
			if len(requestBody.X[i]) != numExamples {
				c.JSON(400, gin.H{"error": "X is not correct. Number of examples is missmatched (some data is missing/too much for some inputs)"})
				return
			}
		}
		for i := 1; i < len(requestBody.Y); i++ {
			if len(requestBody.Y[i]) != numExamples {
				c.JSON(400, gin.H{"error": "Y is not correct. Number of examples is missmatched (some data is missing/too much for some outputs)"})
				return
			}
		}
		

		weights, biases, losses := ml.GradiantDescent(requestBody.X, requestBody.Y, requestBody.Iterations, requestBody.LearningRate, requestBody.MomentumFactor, requestBody.LearningRateDecay, requestBody.Input, requestBody.Hidden, requestBody.Output, requestBody.ActivationFunctions)
		_, outputs := ml.ForwardPropMultiLayer(weights, biases, requestBody.ActivationFunctions, requestBody.X)

		c.JSON(200, TrainResponseBody{Weights: weights, Biases: biases, Losses: losses, Predictions: outputs[len(outputs)-1]})
	})

	r.POST("/train/xor", func(c *gin.Context) {
		X := [][]float64{
			{0, 0, 1, 1},
			{0, 1, 0, 1},
		}
		
		Y := [][]float64{
			{0, 1, 1, 0},
		}

		activationFunctions := []string{"sigmoid", "linear"}
		hiddenLayers := []int{3}
		weights, biases, losses := ml.GradiantDescent(X, Y, 5000, 0.5, 0.001, 0.0005, 2, hiddenLayers, 1, activationFunctions)
		_, outputs := ml.ForwardPropMultiLayer(weights, biases, activationFunctions, X)
		
		c.JSON(200, TrainResponseBody{Weights: weights, Biases: biases, Losses: losses, Predictions: outputs[len(outputs)-1]})
	})

	r.POST("/train/sin", func(c *gin.Context) {
		X := make([][]float64, 1)
		X[0] = make([]float64, 70)
		for i := 0; i < 70; i++ {
			X[0][i] = float64(i) / 10.0
		}

		Y := make([][]float64, 1)
		Y[0] = make([]float64, 70)
		for i := 0; i < 70; i++ {
			Y[0][i] = math.Sin(float64(i) / 10.0)
		}

		hiddenLayers := []int{10, 10}
		activationFunctions := []string{"sigmoid", "sigmoid", "linear"}

		weights, biases, losses := ml.GradiantDescent(X, Y, 30000, 0.05, 0.02, 0.0005, 1, hiddenLayers, 1, activationFunctions)
		_, outputs := ml.ForwardPropMultiLayer(weights, biases, activationFunctions, X)

		c.JSON(200, TrainResponseBody{Weights: weights, Biases: biases, Losses: losses, Predictions: outputs[len(outputs)-1]})
	})

    r.GET("/list", func(c *gin.Context) {
        c.HTML(200, "list.templ", gin.H{"list": []int{1,2,3,4}})
    })

	r.Run() 
}
