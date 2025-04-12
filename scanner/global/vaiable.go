package global

import (
	"sync"
	"sync/atomic"

	"github.com/ethereum/go-ethereum/common"
	"github.com/rrabit42/aidrop-core/common/lib/log"
	"github.com/rrabit42/aidrop-core/types"
)

// filterAddress 등등 다양한 파일에서 변경되는 값들을 관리
type globalVariable struct {
	filterTopics     []common.Hash
	filterTopicsLock sync.RWMutex

	filterAddresses       []common.Address
	filterLockAddressLock sync.RWMutex
	filterAddressesMap    map[common.Address]bool

	LatestReceivedBlockNumber atomic.Uint64
	// FetchSubAvgState          atomic.Bool
	AvgStartChannel chan struct{}

	log log.Logger
}

var GlobalVariable *globalVariable

func init() {
	GlobalVariable = &globalVariable{
		filterAddressesMap:        make(map[common.Address]bool),
		log:                       log.New("global", "variable"),
		LatestReceivedBlockNumber: atomic.Uint64{},
		// FetchSubAvgState:          atomic.Bool{},
		AvgStartChannel: make(chan struct{}),
	}
}

func (g *globalVariable) GetFilterAddresses() []common.Address {
	g.filterLockAddressLock.RLock()
	defer g.filterLockAddressLock.RUnlock()

	return g.filterAddresses
}

func (g *globalVariable) AddFilterAddresses(filterAddresses ...common.Address) {
	g.filterLockAddressLock.RLock()
	defer g.filterLockAddressLock.RUnlock()

	for _, adr := range filterAddresses {
		if _, ok := g.filterAddressesMap[adr]; !ok {
			g.filterAddresses = append(g.filterAddresses, adr)
			g.log.Info("filter address added", "address", adr.String())
		}
	}
}

func (g *globalVariable) GetFilterTopics() []common.Hash {
	g.filterTopicsLock.RLock()
	defer g.filterTopicsLock.RUnlock()

	return g.filterTopics
}

func (g *globalVariable) AddFilterTopics(filterTopics ...common.Hash) {
	g.filterTopicsLock.RLock()
	defer g.filterTopicsLock.RUnlock()

	for _, topic := range filterTopics {
		g.filterTopics = append(g.filterTopics, topic)

		g.log.Info("topic added", "topic", types.TopicsMapper[topic])
	}

}
