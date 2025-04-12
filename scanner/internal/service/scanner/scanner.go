package scanner

import "github.com/rrabit42/aidrop-core/internal/service/chainRpc"

type Scanner struct {
	rpcClient *chainRpc.Client
}

func NewScanner(rpcClient *chainRpc.Client) *Scanner {
	return &Scanner{
		rpcClient: rpcClient,
	}
}
