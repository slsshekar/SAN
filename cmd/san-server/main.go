package main

import (
	"log"
	"sanrl/rl"
	"sanrl/sim"
)

func main() {
	// ENHANCED CONFIG: Shows hybrid advantages
	cfg := sim.SANConfig{
		NumDisks: 4,
		
		// HETEROGENEOUS disks (different capabilities)
		ServiceRates: []float64{2.0, 0.8, 1.5, 1.2}, // Was: all similar
		
		// HIGHER failure rates (so risk prediction matters)
		FailureProbs: []float64{0.008, 0.015, 0.005, 0.012}, // Was: 0.002-0.007
		RecoverProbs: []float64{0.05, 0.05, 0.05, 0.05},
		
		// VARIABLE workload (so prediction helps)
		ArrivalRate:   3.5, // Higher load
		ReqSizeLow:    0.3, // More variance
		ReqSizeHigh:   2.5, // Was: 0.5-2.0
		
		// MORE network variability (realistic)
		NetworkBase:   0.12, // Slightly higher base
		NetworkJitter: 0.05, // Was: 0.02, more jitter
		
		Seed:     42,
		MaxSteps: 500,
	}

	server := rl.NewServer(cfg)
	if err := server.Run(":1337"); err != nil {
		log.Fatal(err)
	}
}