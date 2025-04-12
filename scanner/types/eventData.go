package types

import (
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/shopspring/decimal"
)

// 공통 이벤트 데이터 구조체
type EventData struct {
	BlockNumber uint64
	TxHash      common.Hash
	LogIndex    uint
	EventType   EventType
	Address     common.Hash // 이벤트를 발생시킨 컨트랙트 주소
	Timestamp   int64       // 블록 타임스탬프
}

// ERC20/ERC721 표준 Transfer 이벤트 데이터
type TransferEvent struct {
	EventData
	From  common.Address
	To    common.Address
	Value decimal.Decimal // ERC20: 금액, ERC721: 토큰 ID
}

// ERC20/ERC721 표준 Approval 이벤트 데이터
type ApprovalEvent struct {
	EventData
	Owner   common.Address
	Spender common.Address
	Value   decimal.Decimal // ERC20: 금액, ERC721: 토큰 ID
}

// ERC721/ERC1155 표준 ApprovalForAll 이벤트 데이터
type ApprovalForAllEvent struct {
	EventData
	Owner    common.Address
	Operator common.Address
	Approved bool
}

// ERC1155 표준 TransferSingle 이벤트 데이터
type TransferSingleEvent struct {
	EventData
	Operator common.Address
	From     common.Address
	To       common.Address
	Id       *big.Int
	Value    *big.Int
}

// ERC1155 표준 TransferBatch 이벤트 데이터
type TransferBatchEvent struct {
	EventData
	Operator common.Address
	From     common.Address
	To       common.Address
	Ids      []*big.Int
	Values   []*big.Int
}

// ERC1155 표준 URI 이벤트 데이터
type URIEvent struct {
	EventData
	URI string
	Id  *big.Int
}

// DEX Swap 이벤트 데이터 (Uniswap, SushiSwap 등)
type SwapEvent struct {
	EventData
	Sender     common.Address
	Amount0In  *big.Int
	Amount1In  *big.Int
	Amount0Out *big.Int
	Amount1Out *big.Int
	To         common.Address
}

// UniswapV2 Factory PairCreated 이벤트 데이터
type PairCreatedEvent struct {
	EventData
	Token0    common.Address
	Token1    common.Address
	Pair      common.Address
	PairCount int64
}

// UniswapV2 Pair Sync 이벤트 데이터
type SyncEvent struct {
	EventData
	Reserve0 *big.Int
	Reserve1 *big.Int
}

// Mint 이벤트 데이터
type MintEvent struct {
	EventData
	To     common.Address
	Amount *big.Int
}

// Burn 이벤트 데이터
type BurnEvent struct {
	EventData
	From   common.Address
	Amount *big.Int
}

// Aave/Compound 스타일 Deposit 이벤트 데이터
type DepositEvent struct {
	EventData
	User         common.Address
	Reserve      common.Address
	Amount       *big.Int
	ReferralCode uint16
}

// Aave/Compound 스타일 Withdraw 이벤트 데이터
type WithdrawEvent struct {
	EventData
	User   common.Address
	Amount *big.Int
	To     common.Address
}

// Aave/Compound 스타일 Borrow 이벤트 데이터
type BorrowEvent struct {
	EventData
	User         common.Address
	Reserve      common.Address
	Amount       *big.Int
	InterestRate *big.Int
	ReferralCode uint16
	BorrowRate   *big.Int
}

// Aave/Compound 스타일 Repay 이벤트 데이터
type RepayEvent struct {
	EventData
	User     common.Address
	Repayer  common.Address
	Reserve  common.Address
	Amount   *big.Int
	Interest *big.Int
}

// Aave/Compound 스타일 LiquidationCall 이벤트 데이터
type LiquidationCallEvent struct {
	EventData
	Collateral       common.Address
	DebtAsset        common.Address
	User             common.Address
	DebtToCover      *big.Int
	LiquidatedAmount *big.Int
	Liquidator       common.Address
	ReceiveAToken    bool
}

// ENS NameRegistered 이벤트 데이터
type NameRegisteredEvent struct {
	EventData
	LabelHash [32]byte
	Owner     common.Address
	Cost      *big.Int
}

// ENS NameRenewed 이벤트 데이터
type NameRenewedEvent struct {
	EventData
	LabelHash [32]byte
	Cost      *big.Int
}

// ENS NameTransferred 이벤트 데이터
type NameTransferredEvent struct {
	EventData
	LabelHash [32]byte
	Owner     common.Address
}

// 스테이킹 Staked 이벤트 데이터
type StakedEvent struct {
	EventData
	User   common.Address
	Amount *big.Int
}

// 스테이킹 Unstaked 이벤트 데이터
type UnstakedEvent struct {
	EventData
	User   common.Address
	Amount *big.Int
}

// 스테이킹 RewardClaimed 이벤트 데이터
type RewardClaimedEvent struct {
	EventData
	User   common.Address
	Amount *big.Int
}

// DAO VoteCast 이벤트 데이터
type VoteCastEvent struct {
	EventData
	Voter      common.Address
	ProposalId *big.Int
	Support    bool
	Weight     *big.Int
}

// DAO ProposalCreated 이벤트 데이터 (간소화된 버전)
type ProposalCreatedEvent struct {
	EventData
	ProposalId  *big.Int
	Proposer    common.Address
	Description string
}

// NFT 마켓플레이스 OrderCreated 이벤트 데이터
type OrderCreatedEvent struct {
	EventData
	OrderId   [32]byte
	Maker     common.Address
	AssetAddr common.Address
	TokenId   *big.Int
	Price     *big.Int
}

// NFT 마켓플레이스 OrderCancelled 이벤트 데이터
type OrderCancelledEvent struct {
	EventData
	OrderId [32]byte
}

// NFT 마켓플레이스 OrderMatched 이벤트 데이터
type OrderMatchedEvent struct {
	EventData
	BuyOrderId  [32]byte
	SellOrderId [32]byte
	Maker       common.Address
	Taker       common.Address
	TokenId     *big.Int
	Price       *big.Int
}

// 컨트랙트 Paused 이벤트 데이터
type PausedEvent struct {
	EventData
	Account common.Address
}

// 컨트랙트 Unpaused 이벤트 데이터
type UnpausedEvent struct {
	EventData
	Account common.Address
}

// 컨트랙트 OwnershipTransferred 이벤트 데이터
type OwnershipTransferredEvent struct {
	EventData
	PreviousOwner common.Address
	NewOwner      common.Address
}

// 컨트랙트 RoleGranted 이벤트 데이터
type RoleGrantedEvent struct {
	EventData
	Role    [32]byte
	Account common.Address
	Sender  common.Address
}

// 컨트랙트 RoleRevoked 이벤트 데이터
type RoleRevokedEvent struct {
	EventData
	Role    [32]byte
	Account common.Address
	Sender  common.Address
}
