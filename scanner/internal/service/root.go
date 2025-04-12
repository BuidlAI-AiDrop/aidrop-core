package service

import (
	"path"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	"github.com/rrabit42/aidrop-core/common/util"
	"github.com/rrabit42/aidrop-core/global"
	"github.com/rrabit42/aidrop-core/internal/config"
	"github.com/rrabit42/aidrop-core/internal/service/chainRpc"
	"github.com/rrabit42/aidrop-core/internal/service/scanner"
	"github.com/rrabit42/aidrop-core/internal/service/writer"
	. "github.com/rrabit42/aidrop-core/types"
)

type Service struct {
	client  *chainRpc.Client
	scanner *scanner.Scanner
	writer  *writer.Writer
}

func NewService(cfg *config.Config) {
	s := &Service{
		client: chainRpc.NewClient(cfg.Chain.URL),
	}

	// set log
	if cfg.Log.File.FileName[0] == byte('~') {
		cfg.Log.File.FileName = path.Join(util.HomeDir(), cfg.Log.File.FileName[1:])
	}
	log.SetRoot(cfg.Log.Terminal.Use, cfg.Log.Terminal.Verbosity, cfg.Log.File.Use, cfg.Log.File.Verbosity, cfg.Log.File.FileName)

	if cfg.Chain.Start == 0 {
		// TODO: DB에서 latestBlock 조회
	}

	networkId := s.client.GetNetworkID()
	s.writer = writer.NewWriter(s.client, networkId, cfg.Chain.Name)

	out := make(chan []types.Log, cfg.Chain.ReadingUnit)
	go s.writer.Run(out) // 데이터 처리를 위한 채널 리스닝

	for _, contract := range cfg.Chain.Contracts {
		global.GlobalVariable.AddFilterAddresses(common.HexToAddress(contract))
	}

	for _, t := range Topics {
		global.GlobalVariable.AddFilterTopics(t)
	}

	s.scanner = scanner.NewScanner(cfg, s.client, out)

	select {}
}
