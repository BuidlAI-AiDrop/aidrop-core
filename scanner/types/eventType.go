package types

import (
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

type EventType int

// 표준 및 일반적인 EVM 이벤트 타입 정의
const (
	// ERC-20 / ERC-721
	Transfer EventType = 1 + iota // Transfer(address,address,uint256)
	Approval                      // Approval(address,address,uint256)

	// ERC-721 / ERC-1155
	ApprovalForAll // ApprovalForAll(address,address,bool)

	// ERC-1155
	TransferSingle // TransferSingle(address,address,address,uint256,uint256)
	TransferBatch  // TransferBatch(address,address,address,uint256[],uint256[])
	URI            // URI(string,uint256)

	// 일반적인 DEX 이벤트 (시그니처는 DEX마다 다를 수 있음)
	Swap // Swap(address,uint256,uint256,uint256,uint256,address) - 예시 시그니처

	// 일반적인 토큰 이벤트 (Transfer 이벤트를 사용하는 경우도 많음)
	Mint // Mint(address,uint256) - 예시 시그니처
	Burn // Burn(address,uint256) - 예시 시그니처

	// UniswapV2 이벤트
	PairCreated // PairCreated(address,address,address,uint256)
	Sync        // Sync(uint112,uint112)

	// Lending 프로토콜 이벤트 (Aave, Compound 등)
	Deposit         // Deposit(address,address,uint256,uint16)
	Withdraw        // Withdraw(address,uint256,address)
	Borrow          // Borrow(address,address,uint256,uint256,uint16,uint256)
	Repay           // Repay(address,address,address,uint256,uint256)
	LiquidationCall // LiquidationCall(address,address,address,uint256,uint256,address,bool)

	// ENS 이벤트
	NameRegistered  // NameRegistered(bytes32,address,uint256)
	NameRenewed     // NameRenewed(bytes32,uint256)
	NameTransferred // Transfer(bytes32,address)

	// 스테이킹 / DAO 이벤트
	Staked          // Staked(address,uint256)
	Unstaked        // Unstaked(address,uint256)
	RewardClaimed   // RewardClaimed(address,uint256)
	VoteCast        // VoteCast(address,uint256,bool,uint256)
	ProposalCreated // ProposalCreated(uint256,address,address[],uint256[],string[],bytes[],uint256,uint256,string)

	// NFT 마켓플레이스 이벤트
	OrderCreated   // OrderCreated(bytes32,address,address,uint256,uint256)
	OrderCancelled // OrderCancelled(bytes32)
	OrderMatched   // OrderMatched(bytes32,bytes32,address,address,uint256,uint256)

	// 기타 유틸리티 이벤트
	Paused               // Paused(address)
	Unpaused             // Unpaused(address)
	OwnershipTransferred // OwnershipTransferred(address,address)
	RoleGranted          // RoleGranted(bytes32,address,address)
	RoleRevoked          // RoleRevoked(bytes32,address,address)
)

func (e EventType) String() string {
	switch e {
	case Transfer:
		return "Transfer"
	case Approval:
		return "Approval"
	case ApprovalForAll:
		return "ApprovalForAll"
	case TransferSingle:
		return "TransferSingle"
	case TransferBatch:
		return "TransferBatch"
	case URI:
		return "URI"
	case Swap:
		return "Swap"
	case Mint:
		return "Mint"
	case Burn:
		return "Burn"
	case PairCreated:
		return "PairCreated"
	case Sync:
		return "Sync"
	case Deposit:
		return "Deposit"
	case Withdraw:
		return "Withdraw"
	case Borrow:
		return "Borrow"
	case Repay:
		return "Repay"
	case LiquidationCall:
		return "LiquidationCall"
	case NameRegistered:
		return "NameRegistered"
	case NameRenewed:
		return "NameRenewed"
	case NameTransferred:
		return "NameTransferred"
	case Staked:
		return "Staked"
	case Unstaked:
		return "Unstaked"
	case RewardClaimed:
		return "RewardClaimed"
	case VoteCast:
		return "VoteCast"
	case ProposalCreated:
		return "ProposalCreated"
	case OrderCreated:
		return "OrderCreated"
	case OrderCancelled:
		return "OrderCancelled"
	case OrderMatched:
		return "OrderMatched"
	case Paused:
		return "Paused"
	case Unpaused:
		return "Unpaused"
	case OwnershipTransferred:
		return "OwnershipTransferred"
	case RoleGranted:
		return "RoleGranted"
	case RoleRevoked:
		return "RoleRevoked"
	}
	return "Unknown"
}

// 표준 및 일반적인 이벤트 시그니처 해시
var (
	// ERC-20 / ERC-721
	TransferHash = crypto.Keccak256Hash([]byte("Transfer(address,address,uint256)"))
	ApprovalHash = crypto.Keccak256Hash([]byte("Approval(address,address,uint256)"))

	// ERC-721 / ERC-1155
	ApprovalForAllHash = crypto.Keccak256Hash([]byte("ApprovalForAll(address,address,bool)"))

	// ERC-1155
	TransferSingleHash = crypto.Keccak256Hash([]byte("TransferSingle(address,address,address,uint256,uint256)"))
	TransferBatchHash  = crypto.Keccak256Hash([]byte("TransferBatch(address,address,address,uint256[],uint256[])"))
	URIHash            = crypto.Keccak256Hash([]byte("URI(string,uint256)"))

	// DEX 이벤트
	SwapHash        = crypto.Keccak256Hash([]byte("Swap(address,uint256,uint256,uint256,uint256,address)"))
	PairCreatedHash = crypto.Keccak256Hash([]byte("PairCreated(address,address,address,uint256)"))
	SyncHash        = crypto.Keccak256Hash([]byte("Sync(uint112,uint112)"))

	// 일반적인 토큰 이벤트
	MintHash = crypto.Keccak256Hash([]byte("Mint(address,uint256)"))
	BurnHash = crypto.Keccak256Hash([]byte("Burn(address,uint256)"))

	// Lending 프로토콜 이벤트
	DepositHash         = crypto.Keccak256Hash([]byte("Deposit(address,address,uint256,uint16)"))
	WithdrawHash        = crypto.Keccak256Hash([]byte("Withdraw(address,uint256,address)"))
	BorrowHash          = crypto.Keccak256Hash([]byte("Borrow(address,address,uint256,uint256,uint16,uint256)"))
	RepayHash           = crypto.Keccak256Hash([]byte("Repay(address,address,address,uint256,uint256)"))
	LiquidationCallHash = crypto.Keccak256Hash([]byte("LiquidationCall(address,address,address,uint256,uint256,address,bool)"))

	// ENS 이벤트
	NameRegisteredHash  = crypto.Keccak256Hash([]byte("NameRegistered(bytes32,address,uint256)"))
	NameRenewedHash     = crypto.Keccak256Hash([]byte("NameRenewed(bytes32,uint256)"))
	NameTransferredHash = crypto.Keccak256Hash([]byte("Transfer(bytes32,address)"))

	// 스테이킹 / DAO 이벤트
	StakedHash          = crypto.Keccak256Hash([]byte("Staked(address,uint256)"))
	UnstakedHash        = crypto.Keccak256Hash([]byte("Unstaked(address,uint256)"))
	RewardClaimedHash   = crypto.Keccak256Hash([]byte("RewardClaimed(address,uint256)"))
	VoteCastHash        = crypto.Keccak256Hash([]byte("VoteCast(address,uint256,bool,uint256)"))
	ProposalCreatedHash = crypto.Keccak256Hash([]byte("ProposalCreated(uint256,address,address[],uint256[],string[],bytes[],uint256,uint256,string)"))

	// NFT 마켓플레이스 이벤트
	OrderCreatedHash   = crypto.Keccak256Hash([]byte("OrderCreated(bytes32,address,address,uint256,uint256)"))
	OrderCancelledHash = crypto.Keccak256Hash([]byte("OrderCancelled(bytes32)"))
	OrderMatchedHash   = crypto.Keccak256Hash([]byte("OrderMatched(bytes32,bytes32,address,address,uint256,uint256)"))

	// 기타 유틸리티 이벤트
	PausedHash               = crypto.Keccak256Hash([]byte("Paused(address)"))
	UnpausedHash             = crypto.Keccak256Hash([]byte("Unpaused(address)"))
	OwnershipTransferredHash = crypto.Keccak256Hash([]byte("OwnershipTransferred(address,address)"))
	RoleGrantedHash          = crypto.Keccak256Hash([]byte("RoleGranted(bytes32,address,address)"))
	RoleRevokedHash          = crypto.Keccak256Hash([]byte("RoleRevoked(bytes32,address,address)"))
)

// --> Topic이 추가 된다면, Topics와 TopicsMapper에 모두 추가해준다.
// 추적할 표준 및 일반적인 이벤트 토픽 목록
var Topics = []common.Hash{
	TransferHash,
	ApprovalHash,
	ApprovalForAllHash,
	TransferSingleHash,
	TransferBatchHash,
	URIHash,
	SwapHash,
	MintHash,
	BurnHash,
	PairCreatedHash,
	SyncHash,
	DepositHash,
	WithdrawHash,
	BorrowHash,
	RepayHash,
	LiquidationCallHash,
	NameRegisteredHash,
	NameRenewedHash,
	NameTransferredHash,
	StakedHash,
	UnstakedHash,
	RewardClaimedHash,
	VoteCastHash,
	ProposalCreatedHash,
	OrderCreatedHash,
	OrderCancelledHash,
	OrderMatchedHash,
	PausedHash,
	UnpausedHash,
	OwnershipTransferredHash,
	RoleGrantedHash,
	RoleRevokedHash,
}

// 이벤트 해시와 EventType 매핑
var TopicsMapper = map[common.Hash]EventType{
	TransferHash:             Transfer,
	ApprovalHash:             Approval,
	ApprovalForAllHash:       ApprovalForAll,
	TransferSingleHash:       TransferSingle,
	TransferBatchHash:        TransferBatch,
	URIHash:                  URI,
	SwapHash:                 Swap,
	MintHash:                 Mint,
	BurnHash:                 Burn,
	PairCreatedHash:          PairCreated,
	SyncHash:                 Sync,
	DepositHash:              Deposit,
	WithdrawHash:             Withdraw,
	BorrowHash:               Borrow,
	RepayHash:                Repay,
	LiquidationCallHash:      LiquidationCall,
	NameRegisteredHash:       NameRegistered,
	NameRenewedHash:          NameRenewed,
	NameTransferredHash:      NameTransferred,
	StakedHash:               Staked,
	UnstakedHash:             Unstaked,
	RewardClaimedHash:        RewardClaimed,
	VoteCastHash:             VoteCast,
	ProposalCreatedHash:      ProposalCreated,
	OrderCreatedHash:         OrderCreated,
	OrderCancelledHash:       OrderCancelled,
	OrderMatchedHash:         OrderMatched,
	PausedHash:               Paused,
	UnpausedHash:             Unpaused,
	OwnershipTransferredHash: OwnershipTransferred,
	RoleGrantedHash:          RoleGranted,
	RoleRevokedHash:          RoleRevoked,
}
