package main

import (
	"log"

	"sanrl/rl"
	"sanrl/sim"
)

func main() {
	cfg := sim.SANConfig{
		NumDisks:      4,
		ServiceRates:  []float64{1.5, 0.8, 2.0, 0.6},
		FailureProbs:  []float64{0.002, 0.005, 0.001, 0.007},
		RecoverProbs:  []float64{0.05, 0.05, 0.05, 0.05},
		ArrivalRate:   2.5,
		ReqSizeLow:    0.5,
		ReqSizeHigh:   2.0,
		NetworkBase:   0.1,
		NetworkJitter: 0.02,
		Seed:          42,
		MaxSteps:      500, // episode length
	}

	server := rl.NewServer(cfg)
	if err := server.Run(":1337"); err != nil {
		log.Fatal(err)
	}
}
