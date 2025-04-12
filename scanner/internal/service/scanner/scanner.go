package scanner

import (
	"math/big"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	"github.com/rrabit42/aidrop-core/global"
	"github.com/rrabit42/aidrop-core/internal/config"
	"github.com/rrabit42/aidrop-core/internal/service/chainRpc"
)

type Scanner struct {
	rpcClient   *chainRpc.Client
	startBlock  int64
	readingUnit int64
	log         log.Logger
}

func NewScanner(cfg *config.Config, rpcClient *chainRpc.Client, out chan<- []types.Log) *Scanner {
	s := &Scanner{
		rpcClient:   rpcClient,
		startBlock:  cfg.Chain.Start,
		readingUnit: cfg.Chain.ReadingUnit,
		log:         log.New("module", "service/scanner"),
	}

	go s.loop(out)

	return s
}

func (s *Scanner) loop(out chan<- []types.Log) {
	for {
		filter := ethereum.FilterQuery{
			Addresses: global.GlobalVariable.GetFilterAddresses(),
			Topics:    [][]common.Hash{global.GlobalVariable.GetFilterTopics()},
			FromBlock: big.NewInt(s.startBlock),
		}

		<-time.Tick(1e9)

		latestBlock, err := s.rpcClient.GetLatestBlockNumber()
		if err != nil {
			panic(err)
		}

		if latestBlock < s.startBlock+s.readingUnit-1 {
			filter.ToBlock = big.NewInt(latestBlock)
		} else {
			filter.ToBlock = big.NewInt(s.startBlock + s.readingUnit - 1)
		}

		retry := 1
	SCAN:
		var logs []types.Log

		if logs, err = s.rpcClient.GetLogs(filter); err != nil {
			readingUnit := s.readingUnit / (2 << retry)
			if readingUnit == 0 {
				s.log.Error("Block reading failed", "fromBlock", filter.FromBlock, "toBlock", filter.ToBlock, "top", latestBlock, "err", err)
				return
			}

			filter.ToBlock = big.NewInt(s.startBlock + readingUnit - 1)
			retry++
			goto SCAN

		} else if len(logs) > 0 {
			out <- logs
			s.log.Trace("logs append to channel", "from", filter.FromBlock, "to", filter.ToBlock)
		}

		topScan := latestBlock - filter.ToBlock.Int64()

		s.log.Info("Block scan", "from", filter.FromBlock, "scan", filter.ToBlock, "top", latestBlock, "top-scan", topScan)

		atomic.StoreInt64(&s.startBlock, filter.ToBlock.Int64()+1)
	}
}
