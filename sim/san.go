package sim

import (
	"math/rand"
	"time"
)

// SANConfig holds configuration for the SAN simulator.
type SANConfig struct {
	NumDisks      int
	ServiceRates  []float64 // per-disk service speed
	FailureProbs  []float64 // per-disk failure prob per step
	RecoverProbs  []float64 // per-disk recovery prob per step
	ArrivalRate   float64   // currently unused, but kept for future extension
	ReqSizeLow    float64   // min job size
	ReqSizeHigh   float64   // max job size
	NetworkBase   float64   // base network latency component
	NetworkJitter float64   // random jitter on network latency
	Seed          int64     // RNG seed
	MaxSteps      int       // episode horizon
}

// StepMetrics is returned by each Step().
type StepMetrics struct {
	Step        int
	AvgLatency  float64
	Queues      []float64
	Reward      float64
	Terminated  bool
	LastLatency float64
}

// SAN is a simple, abstract SAN simulator.
type SAN struct {
	cfg SANConfig

	rng *rand.Rand

	step int

	queues       []float64 // queued work per disk
	serviceRates []float64
	alive        []bool

	totalLatency float64
	numJobs      int

	maxSteps int
}

// NewSAN creates a new SAN simulator with the given config.
func NewSAN(cfg SANConfig) *SAN {
	if cfg.NumDisks <= 0 {
		cfg.NumDisks = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 1000
	}

	s := &SAN{
		cfg:          cfg,
		rng:          rand.New(rand.NewSource(cfg.Seed)),
		step:         0,
		queues:       make([]float64, cfg.NumDisks),
		serviceRates: make([]float64, cfg.NumDisks),
		alive:        make([]bool, cfg.NumDisks),
		maxSteps:     maxSteps,
	}

	for i := 0; i < cfg.NumDisks; i++ {
		if i < len(cfg.ServiceRates) && cfg.ServiceRates[i] > 0 {
			s.serviceRates[i] = cfg.ServiceRates[i]
		} else {
			s.serviceRates[i] = 1.0
		}
		s.alive[i] = true
	}
	return s
}

// Observation returns the state vector expected by the Python env:
// [ queues(numDisks), serviceRates(numDisks), aliveMask(numDisks) ].
func (s *SAN) Observation() []float64 {
	n := len(s.queues)
	state := make([]float64, 3*n)

	// Queues
	for i := 0; i < n; i++ {
		state[i] = s.queues[i]
	}

	// Service rates
	for i := 0; i < n; i++ {
		state[n+i] = s.serviceRates[i]
	}

	// Alive mask
	for i := 0; i < n; i++ {
		if s.alive[i] {
			state[2*n+i] = 1.0
		} else {
			state[2*n+i] = 0.0
		}
	}

	return state
}

// Step advances the simulator by one RL decision step.
// `action` = which disk to route the next arriving job to.
func (s *SAN) Step(action int) StepMetrics {
	n := len(s.queues)
	if action < 0 || action >= n {
		action = 0
	}

	s.step++

	// 1) Random disk failures / recoveries
	for i := 0; i < n; i++ {
		if s.alive[i] {
			if i < len(s.cfg.FailureProbs) && s.rng.Float64() < s.cfg.FailureProbs[i] {
				s.alive[i] = false
			}
		} else {
			if i < len(s.cfg.RecoverProbs) && s.rng.Float64() < s.cfg.RecoverProbs[i] {
				s.alive[i] = true
			}
		}
	}

	// 2) Service: each alive disk processes some work
	for i := 0; i < n; i++ {
		if !s.alive[i] {
			continue
		}
		rate := s.serviceRates[i]
		if rate <= 0 {
			continue
		}
		s.queues[i] -= rate
		if s.queues[i] < 0 {
			s.queues[i] = 0
		}
	}

	// 3) Arrival: one job per step (simplified), routed to `action` if alive,
	// otherwise fallback to first alive disk, else job is dropped.
	jobSize := s.cfg.ReqSizeLow
	if s.cfg.ReqSizeHigh > s.cfg.ReqSizeLow {
		jobSize = s.cfg.ReqSizeLow + s.rng.Float64()*(s.cfg.ReqSizeHigh-s.cfg.ReqSizeLow)
	}

	target := action
	if !s.alive[target] {
		found := -1
		for i := 0; i < n; i++ {
			if s.alive[i] {
				found = i
				break
			}
		}
		target = found
	}

	var jobLatency float64
	if target >= 0 {
		// Queueing delay ~ queued work / service rate
		rate := s.serviceRates[target]
		queueDelay := 0.0
		if rate > 0 {
			queueDelay = s.queues[target] / rate
		}

		// Network delay = base + jitter (Gaussian, clamped at 0)
		netDelay := s.cfg.NetworkBase
		if s.cfg.NetworkJitter > 0 {
			j := s.rng.NormFloat64() * s.cfg.NetworkJitter
			netDelay += j
		}
		if netDelay < 0 {
			netDelay = 0
		}

		jobLatency = queueDelay + netDelay
		s.queues[target] += jobSize

		s.totalLatency += jobLatency
		s.numJobs++
	} else {
		// no alive disk, huge penalty
		jobLatency = 1000.0
		s.totalLatency += jobLatency
		s.numJobs++
	}

	avgLatency := 0.0
	if s.numJobs > 0 {
		avgLatency = s.totalLatency / float64(s.numJobs)
	}

	// Reward: minimize latency + queues
	sumQ := 0.0
	for i := 0; i < n; i++ {
		sumQ += s.queues[i]
	}

	// Scaled so values are not huge
	reward := -0.1*jobLatency - 0.001*sumQ

	// Termination condition: fixed horizon episode
	terminated := s.step >= s.maxSteps

	// Copy queues for safe exposure
	qcopy := make([]float64, n)
	copy(qcopy, s.queues)

	return StepMetrics{
		Step:        s.step,
		AvgLatency:  avgLatency,
		Queues:      qcopy,
		Reward:      reward,
		Terminated:  terminated,
		LastLatency: jobLatency,
	}
}
