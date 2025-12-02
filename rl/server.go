package rl

import (
	"bufio"
	"encoding/json"
	"log"
	"net"

	"sanrl/sim"
)

// Request is what we receive from Python: {"action": <int>}
type Request struct {
	Action int `json:"action"`
}

// Response is what we send back: state, reward, terminated flag, and some metrics.
type Response struct {
	State      []float64          `json:"state"`
	Reward     float64            `json:"reward"`
	Terminated bool               `json:"terminated"`
	Metrics    map[string]float64 `json:"metrics"`
}

// Server holds a SANConfig; each new TCP connection gets a fresh SAN instance.
type Server struct {
	cfg sim.SANConfig
}

// NewServer creates a new RL server with the given SAN config.
func NewServer(cfg sim.SANConfig) *Server {
	return &Server{cfg: cfg}
}

// Run starts listening for TCP connections and handles each one as a separate episode.
func (s *Server) Run(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	log.Printf("[SERVER] Listening on %s", addr)

	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Printf("[SERVER] Accept error: %v", err)
			continue
		}
		go s.handleConnection(conn)
	}
}

// handleConnection runs exactly ONE episode per TCP connection.
func (s *Server) handleConnection(conn net.Conn) {
	defer conn.Close()

	log.Printf("[SERVER] Client connected: %s", conn.RemoteAddr().String())

	// Fresh SAN for this episode / connection
	san := sim.NewSAN(s.cfg)

	writer := bufio.NewWriter(conn)
	encoder := json.NewEncoder(writer)
	scanner := bufio.NewScanner(conn)

	// ---- Send initial state ----
	state := san.Observation()
	initialResp := Response{
		State:      state,
		Reward:     0.0,
		Terminated: false,
		Metrics: map[string]float64{
			"step":        0,
			"avg_latency": 0,
		},
	}
	if err := encoder.Encode(&initialResp); err != nil {
		log.Printf("[SERVER] Failed to send initial state: %v", err)
		return
	}
	if err := writer.Flush(); err != nil {
		log.Printf("[SERVER] Flush error (initial): %v", err)
		return
	}
	log.Printf("[SERVER] Initial state sent")

	// ---- Main loop: read actions, step simulation, send back transitions ----
	for {
		if !scanner.Scan() {
			log.Printf("[SERVER] No action received. Connection closed?")
			return
		}

		var req Request
		if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
			log.Printf("[SERVER] Bad action JSON: %v", err)
			return
		}

		metrics := san.Step(req.Action)
		state = san.Observation()

		metricsMap := map[string]float64{
			"step":         float64(metrics.Step),
			"avg_latency":  metrics.AvgLatency,
			"reward":       metrics.Reward,
			"last_latency": metrics.LastLatency,
		}
		if len(metrics.Queues) > 0 {
			metricsMap["queue_0"] = metrics.Queues[0]
		}

		resp := Response{
			State:      state,
			Reward:     metrics.Reward,
			Terminated: metrics.Terminated,
			Metrics:    metricsMap,
		}

		if err := encoder.Encode(&resp); err != nil {
			log.Printf("[SERVER] Encode error: %v", err)
			return
		}
		if err := writer.Flush(); err != nil {
			log.Printf("[SERVER] Flush error: %v", err)
			return
		}

		if metrics.Terminated {
			log.Printf("[SERVER] Episode finished at step %d", metrics.Step)
			return // end of episode; connection closes
		}
	}
}
